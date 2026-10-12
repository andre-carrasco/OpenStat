"""Result: what an analysis returns — text, tables, figures and columns to store."""
from __future__ import annotations

import base64
import html
import io
from dataclasses import dataclass, field

import pandas as pd

from openstat.i18n import tr

CSS = (
    "body{font-family:Segoe UI,Arial,sans-serif;font-size:10pt;}"
    "h2{font-size:13pt;color:#1f4e79;margin:14px 0 4px 0;}"
    "h3{font-size:10.5pt;margin:10px 0 2px 0;}"
    "th{text-align:right;border-bottom:1px solid #666;padding:2px 10px;}"
    "td{text-align:right;padding:2px 10px;}"
    "td.l,th.l{text-align:left;}"
)


def format_number(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "*"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int) or (hasattr(value, "dtype") and getattr(value.dtype, "kind", "") in "iu"):
        return str(int(value))
    if isinstance(value, float) or (hasattr(value, "dtype") and getattr(value.dtype, "kind", "") == "f"):
        value = float(value)
        if value != value:
            return "*"
        if value != 0 and (abs(value) >= 1e7 or abs(value) < 1e-4):
            return f"{value:.{digits}e}"
        return f"{value:.{digits}f}"
    return html.escape(str(value))


@dataclass
class Result:
    title: str
    blocks: list = field(default_factory=list)
    stored: dict = field(default_factory=dict)
    worksheets: list = field(default_factory=list)
    formulas: dict = field(default_factory=dict)

    def heading(self, text: str) -> "Result":
        self.blocks.append(("heading", text))
        return self

    def text(self, text: str) -> "Result":
        self.blocks.append(("text", text))
        return self

    def table(self, df: pd.DataFrame, title: str | None = None, digits: int = 4, index: bool = True) -> "Result":
        self.blocks.append(("table", (df, title, digits, index)))
        return self

    def pre(self, text: str) -> "Result":
        """Monospaced text, for character graphics and listings."""
        self.blocks.append(("pre", text))
        return self

    def worksheet(self, name: str, df: pd.DataFrame) -> "Result":
        """Ask the application to open `df` as a new worksheet."""
        self.worksheets.append((name, df))
        return self

    def figure(self, fig) -> "Result":
        self.blocks.append(("figure", fig))
        return self

    def store(self, name: str, values) -> "Result":
        self.stored[name] = list(values)
        return self

    def formula(self, name: str, expression: str) -> "Result":
        """Ask the application to attach a formula to column `name`; an empty one detaches it."""
        self.formulas[name] = expression
        return self

    def figures(self) -> list:
        return [payload for kind, payload in self.blocks if kind == "figure"]

    def to_html(self, anchor: str | None = None) -> str:
        """HTML fragment; figures are embedded as base64 PNG."""
        parts = []
        name = f' id="{anchor}"' if anchor else ""
        parts.append(f'<h2{name}><a name="{anchor or ""}"></a>{html.escape(tr(self.title))}</h2>')
        for kind, payload in self.blocks:
            if kind == "heading":
                parts.append(f"<h3>{html.escape(tr(payload))}</h3>")
            elif kind == "text":
                parts.append(f"<p>{html.escape(payload).replace(chr(10), '<br>')}</p>")
            elif kind == "pre":
                parts.append(f"<pre>{html.escape(payload)}</pre>")
            elif kind == "table":
                parts.append(_table_html(*payload))
            elif kind == "figure":
                buf = io.BytesIO()
                payload.savefig(buf, format="png", dpi=100)
                data = base64.b64encode(buf.getvalue()).decode("ascii")
                parts.append(f'<p><img src="data:image/png;base64,{data}"></p>')
        return "\n".join(parts)


def _table_html(df: pd.DataFrame, title, digits: int, index: bool) -> str:
    out = []
    if title:
        out.append(f"<h3>{html.escape(tr(title))}</h3>")
    out.append('<table cellspacing="0">')
    head = ['<th class="l">' + html.escape(tr(str(df.index.name or ""))) + "</th>"] if index else []
    head += [f"<th>{html.escape(tr(str(c)))}</th>" for c in df.columns]
    out.append("<tr>" + "".join(head) + "</tr>")
    for label, row in zip(df.index, df.itertuples(index=False)):
        cells = [f'<td class="l">{html.escape(_row_label(label))}</td>'] if index else []
        cells += [f"<td>{format_number(v, digits)}</td>" for v in row]
        out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def _row_label(label) -> str:
    """Translate a row label, keeping the indentation used for nested rows."""
    text = str(label)
    stripped = text.strip()
    return text.replace(stripped, tr(stripped)) if stripped else text


def wrap_document(body: str, title: str = "OpenStat") -> str:
    return (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )
