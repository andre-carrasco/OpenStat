"""Registry of analyses. Menus and dialogs are generated from these declarations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# Param kinds:
#   columns  one or more worksheet columns      -> list[str]
#   column   exactly one worksheet column       -> str (or None when optional)
#   number   floating point value               -> float
#   int      integer value                      -> int
#   choice   one of `choices`                   -> str
#   bool     checkbox                           -> bool
#   text     free text                          -> str


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
