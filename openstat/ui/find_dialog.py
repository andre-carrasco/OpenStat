from __future__ import annotations

import re

from PySide6.QtWidgets import (QCheckBox, QDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QVBoxLayout)

from openstat.core.worksheet import Worksheet
from openstat.i18n import tr


def _matches(cell: str, text: str, match_case: bool, whole_cell: bool) -> bool:
    if not match_case:
        cell, text = cell.lower(), text.lower()
    return cell == text if whole_cell else text in cell


def find_next(ws: Worksheet, text: str, start: tuple[int, int], match_case: bool, whole_cell: bool,
              columns: list[int] | None) -> tuple[int, int] | None:
    """First matching cell after `start` in row order, wrapping once around the sheet."""
    n_cols = ws.ncols
    total = ws.nrows * n_cols
    first = start[0] * n_cols + start[1] + 1
    for k in range(total):
        row, col = divmod((first + k) % total, n_cols)
        if (columns is None or col in columns) and _matches(ws.get_text(row, col), text, match_case, whole_cell):
            return row, col
    return None


class FindDialog(QDialog):
    def __init__(self, window, replace: bool = False):
        super().__init__(window)
        self.window_ = window
        self.find_text = QLineEdit()
        self.replace_text = QLineEdit()
        self.match_case = QCheckBox(tr("Match case"))
        self.whole_cell = QCheckBox(tr("Match entire cell"))
        self.selected_only = QCheckBox(tr("Selected columns only"))
        self.status = QLabel()
        self.replace_label = QLabel(tr("Replace with"))
        self.replace_button = QPushButton(tr("Replace"))
        self.replace_all_button = QPushButton(tr("Replace All"))
        next_button = QPushButton(tr("Find Next"))
        next_button.setDefault(True)
        next_button.clicked.connect(self.find_next)
        self.replace_button.clicked.connect(self.replace_one)
        self.replace_all_button.clicked.connect(self.replace_all)

        form = QFormLayout()
        form.addRow(tr("Find what"), self.find_text)
        form.addRow(self.replace_label, self.replace_text)
        buttons = QHBoxLayout()
        for button in (next_button, self.replace_button, self.replace_all_button):
            buttons.addWidget(button)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        for box in (self.match_case, self.whole_cell, self.selected_only, self.status):
            layout.addWidget(box)
        layout.addLayout(buttons)
        self.set_replace(replace)

    def set_replace(self, replace: bool) -> None:
        self.setWindowTitle(tr("Replace...") if replace else tr("Find..."))
        for widget in (self.replace_label, self.replace_text, self.replace_button, self.replace_all_button):
            widget.setVisible(replace)

    def _columns(self, view) -> list[int] | None:
        if not self.selected_only.isChecked():
            return None
        return sorted({i.column() for i in view.selectedIndexes()})

    def _options(self) -> tuple[bool, bool]:
        return self.match_case.isChecked(), self.whole_cell.isChecked()

    def find_next(self) -> bool:
        view = self.window_.current_view()
        current = view.currentIndex()
        start = (current.row(), current.column()) if current.isValid() else (-1, -1)
        found = find_next(view.ws, self.find_text.text(), start, *self._options(), self._columns(view))
        self.status.setText("" if found else tr("Not found"))
        if found:
            view.setCurrentIndex(view.model_.index(*found))
        return found is not None

    def _new_text(self, cell: str) -> str:
        if self.whole_cell.isChecked():
            return self.replace_text.text()
        flags = 0 if self.match_case.isChecked() else re.IGNORECASE
        return re.sub(re.escape(self.find_text.text()), lambda _m: self.replace_text.text(), cell, flags=flags)

    def replace_one(self) -> None:
        view = self.window_.current_view()
        current = view.currentIndex()
        if current.isValid() and _matches(view.ws.get_text(current.row(), current.column()),
                                          self.find_text.text(), *self._options()):
            row, col = current.row(), current.column()
            new = self._new_text(view.ws.get_text(row, col))
            view.change("Replace", lambda: view.ws.set_cell(row, col, new))
            view.refresh()
        self.find_next()

    def replace_all(self) -> None:
        view = self.window_.current_view()
        ws, columns = view.ws, self._columns(view)
        hits = [(r, c) for r in range(ws.nrows) for c in range(ws.ncols)
                if (columns is None or c in columns)
                and _matches(ws.get_text(r, c), self.find_text.text(), *self._options())]
        def apply():
            for r, c in hits:
                ws.set_cell(r, c, self._new_text(ws.get_text(r, c)))
        if hits:
            view.change("Replace", apply)
            view.refresh()
        self.status.setText(tr("{n} cells replaced").format(n=len(hits)) if hits else tr("Not found"))
