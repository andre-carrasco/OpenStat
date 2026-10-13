"""Registry of analyses. Menus and dialogs are generated from these declarations."""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Param kinds:
#   columns  one or more worksheet columns      -> list[str]
#   column   exactly one worksheet column       -> str (or None when optional)
#   number   floating point value               -> float
#   int      integer value                      -> int
#   choice   one of `choices`                   -> str
#   bool     checkbox                           -> bool
#   text     free text                          -> str
#   worksheet  one worksheet of the project     -> Worksheet (the console also takes its name)


@dataclass
class Param:
    name: str
    label: str
    kind: str
    default: Any = None
    choices: list[str] = field(default_factory=list)
    numeric: bool = True        # column kinds: restrict to numeric columns
    optional: bool = False
    minimum: float | None = None
    maximum: float | None = None


@dataclass
class Analysis:
    id: str
    menu: tuple[str, ...]
    title: str
    params: list[Param]
    func: Callable


_REGISTRY: dict[str, Analysis] = {}


def analysis(id: str, menu: tuple[str, ...], title: str, params: list[Param]):
    def decorator(func):
        if id in _REGISTRY:
            raise ValueError(f"Duplicate analysis id: {id}")
        _REGISTRY[id] = Analysis(id, tuple(menu), title, params, func)
        return func
    return decorator


def format_call(a: Analysis, values: dict) -> str:
    """The console command that runs analysis `a` with these values; defaults are left out."""
    parts = []
    for p in a.params:
        value = values.get(p.name)
        if p.kind == "worksheet":
            value = getattr(value, "name", value)
        if value is None or value == [] or (p.kind not in ("column", "columns", "worksheet") and value == p.default):
            continue
        parts.append(f"{p.name}={value!r}")
    return f"run({', '.join([repr(a.id)] + parts)})"


def all_analyses() -> list[Analysis]:
    load_builtin()
    return list(_REGISTRY.values())


def get(id: str) -> Analysis:
    load_builtin()
    return _REGISTRY[id]


def load_builtin() -> None:
    """Import every module that registers analyses."""
    import openstat.graphs  # noqa: F401
    import openstat.stats  # noqa: F401


def plugins_dir() -> Path:
    return Path.home() / ".openstat" / "plugins"


def load_plugins(directory: Path | None = None) -> list[str]:
    """Import every .py file in the plugins folder; returns one message per plugin that failed."""
    load_builtin()
    directory = directory or plugins_dir()
    errors = []
    if not directory.is_dir():
        return errors
    for path in sorted(directory.glob("*.py")):
        try:
            spec = importlib.util.spec_from_file_location(f"openstat_plugin_{path.stem}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as exc:  # a broken plugin must not stop the application
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
    return errors
