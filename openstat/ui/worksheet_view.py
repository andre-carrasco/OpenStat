"""Editable grid over a Worksheet."""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QColor, QGuiApplication, QKeySequence, QUndoStack
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout, QInputDialog, QLineEdit, QMenu,
                               QMessageBox, QTableView)

from openstat.core.formatting import COLORS, RULES, column_colors, make_rule
from openstat.core.worksheet import Worksheet, changed_parts
from openstat.i18n import tr
from openstat.ui.undo import SnapshotCommand

EXTRA_ROWS = 100
EXTRA_COLS = 20


class FormatDialog(QDialog):
    """Asks for one conditional formatting rule."""

    def __init__(self, column: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{tr('Conditional Formatting')}: {column}")
        self.kind, self.value, self.color = QComboBox(), QLineEdit(), QComboBox()
        for name in RULES:
            self.kind.addItem(tr(name), name)
        for name in COLORS:
            self.color.addItem(tr(name), name)
        form = QFormLayout(self)
        form.addRow(tr("Highlight cells that are") + ":", self.kind)
        form.addRow(tr("Value") + ":", self.value)
        form.addRow(tr("Color") + ":", self.color)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText(tr("OK"))
        buttons.button(QDialogButtonBox.Cancel).setText(tr("Cancel"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def rule(self) -> dict:
        return make_rule(self.kind.currentData(), self.value.text(), self.color.currentData())


class WorksheetModel(QAbstractTableModel):
    def __init__(self, worksheet: Worksheet, change: Callable, parent=None):
        super().__init__(parent)
        self.ws = worksheet
        self.change = change   # WorksheetView.change, which makes each change one undo step
        self._colors: dict[str, list] = {}   # conditional formatting of each column, until the next change
        self.modelAboutToBeReset.connect(lambda: self._colors.clear())

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self.ws.nrows + EXTRA_ROWS

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self.ws.ncols + EXTRA_COLS

    def background(self, row: int, column: int) -> str | None:
        if row >= self.ws.nrows or column >= self.ws.ncols:
            return None
        name = self.ws.df.columns[column]
        if name not in self.ws.formats:
            return None
        if name not in self._colors:
            self._colors[name] = column_colors(self.ws, name)
        return self._colors[name][row]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role in (Qt.DisplayRole, Qt.EditRole):
            return self.ws.get_text(index.row(), index.column())
        if role == Qt.BackgroundRole:
            color = self.background(index.row(), index.column())
            return QColor(color) if color else None
        if role == Qt.ForegroundRole and self.background(index.row(), index.column()):
            return QColor("black")   # the highlight colors are light, whatever the theme
        if role == Qt.TextAlignmentRole and index.column() < self.ws.ncols:
            numeric = self.ws.is_numeric(self.ws.df.columns[index.column()])
            return int((Qt.AlignRight if numeric else Qt.AlignLeft) | Qt.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.ToolTipRole and orientation == Qt.Horizontal and section < self.ws.ncols:
            name = self.ws.df.columns[section]
            if name in self.ws.formulas:
                error = self.ws.formula_errors.get(name)
                return f"= {self.ws.formulas[name]}" + (f"\n{tr(error)}" if error else "")
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Vertical:
            return str(section + 1)
        label = f"C{section + 1}"
        if section < self.ws.ncols:
            name = self.ws.df.columns[section]
            if self.ws.is_datetime(name):
                label += "-D"
            elif not self.ws.is_numeric(name):
                label += "-T"
            if name in self.ws.formulas:
                label += " ="
            if name != f"C{section + 1}":
                return f"{label}\n{name}"
        return f"{label}\n"

    def flags(self, index):
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable

    def setData(self, index, value, role=Qt.EditRole):
        if role != Qt.EditRole or not index.isValid():
            return False
        self.set_cells([(index.row(), index.column(), value)])
        return True

    def set_cells(self, cells, text: str = "Typing") -> None:
        def apply():
            for row, col, value in cells:
                self.ws.set_cell(row, col, value)
        self.change(text, apply)
        self.refresh()

    def refresh(self) -> None:
        self.beginResetModel()
        self.endResetModel()


class WorksheetView(QTableView):
    def __init__(self, worksheet: Worksheet, parent=None):
        super().__init__(parent)
        self.ws = worksheet
        self.undo_stack = QUndoStack(self)
        self.undo_stack.setUndoLimit(50)
        self._editing = False
        self.model_ = WorksheetModel(worksheet, self.change, self)
        self.setModel(self.model_)
        self.horizontalHeader().setDefaultSectionSize(90)
        self.verticalHeader().setDefaultSectionSize(22)
        self.horizontalHeader().sectionDoubleClicked.connect(self.rename_column)
        self.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.horizontalHeader().customContextMenuRequested.connect(self._header_menu)

    def refresh(self) -> None:
        self.model_.refresh()

    def change(self, text: str, action: Callable[[], object]) -> object:
        """Run `action` as one undo step named `text`; a call inside another joins the outer step."""
        if self._editing:
            return action()
        before = self.ws.snapshot()
        self._editing = True
        try:
            result = action()
        except Exception:
            self.ws.restore(before)
            self.refresh()
            raise
        finally:
            self._editing = False
        old, new = changed_parts(before, self.ws.snapshot())
        if old["data"] or new["data"] or old != new:
            self.undo_stack.push(SnapshotCommand(self, tr(text), old, new))
        return result

    def _header_menu(self, position) -> None:
        section = self.horizontalHeader().logicalIndexAt(position)
        if section < 0 or section >= self.ws.ncols:
            return
        name = self.ws.df.columns[section]
        menu = QMenu(self)
        menu.addAction(tr("Rename column") + "...", lambda: self.rename_column(section))
        menu.addAction(tr("Conditional Formatting") + "...", lambda: self.add_format(name))
        clear = menu.addAction(tr("Clear Conditional Formatting"), lambda: self.set_formats(name, []))
        clear.setEnabled(name in self.ws.formats)
        menu.exec(self.horizontalHeader().mapToGlobal(position))

    def add_format(self, name: str) -> None:
        dialog = FormatDialog(name, self)
        if dialog.exec() == QDialog.Accepted:
            try:
                rule = dialog.rule()
            except ValueError as exc:
                QMessageBox.warning(self, tr("Error"), tr(str(exc)))
                return
            self.set_formats(name, self.ws.formats.get(name, []) + [rule])

    def set_formats(self, name: str, rules: list[dict]) -> None:
        def apply():
            if rules:
                self.ws.formats[name] = rules
            else:
                self.ws.formats.pop(name, None)
        self.change("Conditional Formatting", apply)
        self.refresh()

    def rename_column(self, section: int) -> None:
        current = self.ws.df.columns[section] if section < self.ws.ncols else f"C{section + 1}"
        name, ok = QInputDialog.getText(self, tr("Rename column"), tr("Column name:"), text=current)
        if ok:
            def apply():
                if section >= self.ws.ncols:
                    self.ws.ensure_size(self.ws.nrows, section + 1)
                self.ws.rename_column(section, name)
            try:
                self.change("Rename column", apply)
            except ValueError as exc:
                QMessageBox.warning(self, tr("Error"), str(exc))
        self.refresh()

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Copy):
            self.copy_selection()
        elif event.matches(QKeySequence.Paste):
            self.paste()
        elif event.key() in (Qt.Key_Delete, Qt.Key_Backspace) and self.state() != QTableView.EditingState:
            self.model_.set_cells([(i.row(), i.column(), "") for i in self.selectedIndexes()])
        else:
            super().keyPressEvent(event)

    def copy_selection(self) -> None:
        indexes = self.selectedIndexes()
        if not indexes:
            return
        rows = sorted({i.row() for i in indexes})
        cols = sorted({i.column() for i in indexes})
        lines = ["\t".join(self.ws.get_text(r, c) for c in cols) for r in rows]
        QGuiApplication.clipboard().setText("\n".join(lines))

    def paste(self) -> None:
        text = QGuiApplication.clipboard().text()
        start = self.currentIndex()
        if not text or not start.isValid():
            return
        cells = []
        for dr, line in enumerate(text.rstrip("\n").split("\n")):
            for dc, value in enumerate(line.rstrip("\r").split("\t")):
                cells.append((start.row() + dr, start.column() + dc, value))
        self.model_.set_cells(cells, "Paste")
