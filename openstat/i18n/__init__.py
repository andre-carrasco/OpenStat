"""String table. Source strings are English; other languages map English -> translation."""
from __future__ import annotations

import json
import os
from pathlib import Path

_table: dict[str, str] = {}
_language = "en"


def set_language(lang: str) -> None:
    global _table, _language
    _language = lang
    path = Path(__file__).with_name(f"{lang}.json")
    _table = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def language() -> str:
    return _language


def detect_language() -> str:
    lang = os.environ.get("OPENSTAT_LANG")
    if not lang:
        import locale
        lang = (locale.getlocale()[0] or "en")
    return "pt_BR" if lang.lower().startswith("pt") else "en"


def tr(text: str) -> str:
    return _table.get(text, text)
