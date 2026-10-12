"""Project: worksheets plus the output history, saved as a .ostat zip archive."""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from openstat.core.worksheet import Worksheet

FORMAT_VERSION = 1


@dataclass
class OutputItem:
    title: str
    html: str


@dataclass
class Project:
    worksheets: list[Worksheet] = field(default_factory=list)
    outputs: list[OutputItem] = field(default_factory=list)
    path: Path | None = None

    def new_worksheet(self) -> Worksheet:
        names = {ws.name for ws in self.worksheets}
        k = 1
        while f"Worksheet {k}" in names:
            k += 1
        ws = Worksheet(f"Worksheet {k}")
        self.worksheets.append(ws)
        return ws

    def save(self, path: str | Path) -> None:
        path = Path(path)
        manifest = {
            "format": FORMAT_VERSION,
            "worksheets": [ws.name for ws in self.worksheets],
            "outputs": [o.title for o in self.outputs],
        }
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            for i, ws in enumerate(self.worksheets):
                buf = io.BytesIO()
                ws.df.to_parquet(buf, index=False)
                z.writestr(f"worksheets/{i}.parquet", buf.getvalue())
            for i, out in enumerate(self.outputs):
                z.writestr(f"outputs/{i}.html", out.html)
        self.path = path

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        path = Path(path)
        project = cls(path=path)
        with zipfile.ZipFile(path) as z:
            manifest = json.loads(z.read("manifest.json").decode("utf-8"))
            if manifest.get("format", 0) > FORMAT_VERSION:
                raise ValueError("Project file was created by a newer version of OpenStat")
            for i, name in enumerate(manifest["worksheets"]):
                df = pd.read_parquet(io.BytesIO(z.read(f"worksheets/{i}.parquet")))
                project.worksheets.append(Worksheet(name, df))
            for i, title in enumerate(manifest["outputs"]):
                project.outputs.append(OutputItem(title, z.read(f"outputs/{i}.html").decode("utf-8")))
        return project
