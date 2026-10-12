"""Command line: a Python console bound to the current worksheet."""
from __future__ import annotations

import code
import contextlib
import io

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QLineEdit, QPlainTextEdit, QVBoxLayout, QWidget

from openstat.core import registry
from openstat.core.worksheet import normalize_frame
from openstat.i18n import tr

BANNER = ["ws: current worksheet   df: its data (pandas DataFrame)   np, pd",
          "run(id, **parameters): run an analysis      analyses(text=''): list analysis ids"]


class HistoryLine(QLineEdit):
    """Input line with command history on the up and down arrows."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.history: list[str] = []
        self.position = 0

    def remember(self, text: str) -> None:
        if text.strip() and (not self.history or self.history[-1] != text):
            self.history.append(text)
        self.position = len(self.history)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Up and self.history:
            self.position = max(0, self.position - 1)
            self.setText(self.history[self.position])
        elif event.key() == Qt.Key_Down and self.history:
            self.position = min(len(self.history), self.position + 1)
            self.setText(self.history[self.position] if self.position < len(self.history) else "")
        else:
            super().keyPressEvent(event)


class Console(QWidget):
    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window_ = window
        self.namespace = {"np": np, "pd": pd, "run": self.run, "analyses": self.analyses}
        self.interpreter = code.InteractiveInterpreter(self.namespace)
        self.pending: list[str] = []
        font = QFont("Consolas")
        font.setStyleHint(QFont.Monospace)
        self.log = QPlainTextEdit(readOnly=True)
        self.log.setFont(font)
        self.log.setPlainText("\n".join(tr(line) for line in BANNER))
        self.line = HistoryLine()
        self.line.setFont(font)
        self.line.setPlaceholderText(tr("Python command, e.g.  df['C3'] = df['C1'] * 2"))
        self.line.returnPressed.connect(self._enter)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.log)
        layout.addWidget(self.line)

    # ----- functions available in the console -----

    def run(self, analysis_id: str, **parameters) -> None:
        """Run a registered analysis on the current worksheet and show its output."""
        ws = self.window_.current_worksheet()
        self.window_.add_result(registry.get(analysis_id).func(ws, **parameters), ws)

    def analyses(self, text: str = "") -> None:
        for a in registry.all_analyses():
            if text.lower() in a.id.lower() or text.lower() in a.title.lower():
                names = ", ".join(p.name for p in a.params)
                print(f"{a.id}({names})  -  {a.title}")

    # ----- execution -----

    def _enter(self) -> None:
        text = self.line.text()
        self.line.clear()
        self.line.remember(text)
        self.execute(text)

    def execute(self, text: str) -> str:
        """Run one line; returns what it printed."""
        ws = self.window_.current_worksheet()
        self.namespace["ws"], self.namespace["df"] = ws, ws.df
        self.log.appendPlainText(("... " if self.pending else ">>> ") + text)
        self.pending.append(text)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            incomplete = self.interpreter.runsource("\n".join(self.pending), "<console>")
        if not incomplete:
            self.pending = []
        output = buffer.getvalue().rstrip()
        if output:
            self.log.appendPlainText(output)
        new_df = self.namespace.get("df")
        if isinstance(new_df, pd.DataFrame):
            ws.df = normalize_frame(new_df)   # whatever the command produced, in worksheet form
        self.window_.worksheets_changed()
        return output
