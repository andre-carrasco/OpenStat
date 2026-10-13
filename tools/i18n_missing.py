"""List strings used in menus, dialogs and results that have no pt-BR translation.

Usage: python tools/i18n_missing.py
"""
import ast
import json
from pathlib import Path

from openstat.core import assistant, formatting, registry

ROOT = Path(__file__).resolve().parents[1] / "openstat"
TABLE = json.loads((ROOT / "i18n" / "pt_BR.json").read_text(encoding="utf-8"))


def source_strings() -> set[str]:
    """Literal arguments of tr(...) and Result(...) calls, and of result headings."""
    found = set()
    for path in ROOT.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call) and node.exc.args:
                message = node.exc.args[0]   # error messages shown to the user
                if isinstance(message, ast.Constant) and isinstance(message.value, str):
                    found.add(message.value)
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            first = node.args[0] if node.args else None
            if name in ("tr", "Result", "heading") and isinstance(first, ast.Constant) and isinstance(first.value, str):
                found.add(first.value)
            for kw in node.keywords:  # table titles
                if kw.arg == "title" and name in ("table", "analysis") and isinstance(kw.value, ast.Constant):
                    found.add(kw.value.value)
    return found


def registry_strings() -> set[str]:
    """Strings that reach tr() through data rather than as literals."""
    found = assistant.strings() | set(formatting.RULES) | set(formatting.COLORS)
    for a in registry.all_analyses():
        found.add(a.title)
        found.update(a.menu)
        for p in a.params:
            found.add(p.label)
            found.update(p.choices)
    return found


if __name__ == "__main__":
    missing = sorted((source_strings() | registry_strings()) - set(TABLE))
    print(json.dumps(missing, ensure_ascii=False, indent=0))
