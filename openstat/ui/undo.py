"""Undo steps for worksheet edits."""
from __future__ import annotations

from PySide6.QtGui import QUndoCommand


class SnapshotCommand(QUndoCommand):
    """One edit, kept as the worksheet state before and after it."""

    def __init__(self, view, text: str, before: dict, after: dict):
        super().__init__(text)
        self.view, self.before, self.after = view, before, after
        self.skip_redo = True   # pushed after the edit ran, so the first redo has nothing to do

    def undo(self) -> None:
        self.view.ws.restore(self.before)
        self.view.refresh()

    def redo(self) -> None:
        if self.skip_redo:
            self.skip_redo = False
            return
        self.view.ws.restore(self.after)
        self.view.refresh()
