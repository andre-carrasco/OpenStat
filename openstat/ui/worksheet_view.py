"""Editable grid over a Worksheet."""
from __future__ import annotations

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QGuiApplication, QKeySequence
from PySide6.QtWidgets import QInputDialog, QMessageBox, QTableView

from openstat.core.worksheet import Worksheet
from openstat.i18n import tr

EXTRA_ROWS = 100
EXTRA_COLS = 20


class WorksheetModel(QAbstractTableModel):
    def __init__(self, worksheet: Worksheet, parent=None):
        super().__init__(parent)
        self.ws = worksheet

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self.ws.nrows + EXTRA_ROWS

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self.ws.ncols + EXTRA_COLS

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role in (Qt.DisplayRole, Qt.EditRole):
            return self.ws.get_text(index.row(), index.column())
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

    def set_cells(self, cells) -> None:
        self.beginResetModel()
        for row, col, text in cells:
            self.ws.set_cell(row, col, text)
        self.endResetModel()

    def refresh(self) -> None:
        self.beginResetModel()
        self.endResetModel()


class WorksheetView(QTableView):
    def __init__(self, worksheet: Worksheet, parent=None):
        super().__init__(parent)
        self.ws = worksheet
        self.model_ = WorksheetModel(worksheet, self)
        self.setModel(self.model_)
        self.horizontalHeader().setDefaultSectionSize(90)
        self.verticalHeader().setDefaultSectionSize(22)
        self.horizontalHeader().sectionDoubleClicked.connect(self.rename_column)

    def refresh(self) -> None:
        self.model_.refresh()

    def rename_column(self, section: int) -> None:
        if section >= self.ws.ncols:
            self.ws.ensure_size(self.ws.nrows, section + 1)
        current = self.ws.df.columns[section]
        name, ok = QInputDialog.getText(self, tr("Rename column"), tr("Column name:"), text=current)
        if ok:
            try:
                self.ws.rename_column(section, name)
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
        self.model_.set_cells(cells)
