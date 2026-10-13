"""Every string shown in menus and dialogs must have a pt-BR translation."""
import json
from pathlib import Path

from openstat.core import assistant, formatting, registry

TABLE = json.loads((Path(registry.__file__).parents[1] / "i18n" / "pt_BR.json").read_text(encoding="utf-8"))


def test_registry_strings_translated():
    strings = assistant.strings() | set(formatting.RULES) | set(formatting.COLORS)
    for a in registry.all_analyses():
        strings.add(a.title)
        strings.update(a.menu)
        for p in a.params:
            strings.add(p.label)
            strings.update(p.choices)
    missing = sorted(s for s in strings if s not in TABLE)
    assert not missing, "Missing pt_BR translations:\n" + "\n".join(missing)
