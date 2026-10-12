"""Merge a JSON file of {english: portuguese} pairs into openstat/i18n/pt_BR.json.

Usage: python tools/i18n_merge.py new_strings.json
"""
import json
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "openstat" / "i18n" / "pt_BR.json"

if __name__ == "__main__":
    table = json.loads(TARGET.read_text(encoding="utf-8"))
    new = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    table.update(new)
    TARGET.write_text(json.dumps(table, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(new)} merged, {len(table)} total")
