"""Editable grid over a Worksheet."""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QColor, QGuiApplication, QKeySequence, QUndoStack
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QInputDialog, QLineEdit,
                               QListWidget, QListWidgetItem, QMenu, QMessageBox, QPushButton, QTableView, QVBoxLayout)

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


class MoveColumnsDialog(QDialog):
    """Asks which columns to move and where."""

    def __init__(self, columns: list[str], selected: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Move Columns"))
        self.columns = QListWidget()
        for name in columns:
            item = QListWidgetItem(name, self.columns)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in selected else Qt.Unchecked)
        self.where, self.anchor = QComboBox(), QComboBox()
        for key, label in (("start", "Start"), ("end", "End"), ("before", "Before"), ("after", "After")):
            self.where.addItem(tr(label), key)
        self.anchor.addItems(columns)
        self.where.currentIndexChanged.connect(
            lambda: self.anchor.setEnabled(self.where.currentData() in ("before", "after")))
        self.anchor.setEnabled(False)
        form = QFormLayout(self)
        form.addRow(tr("Columns") + ":", self.columns)
        form.addRow(tr("Move to") + ":", self.where)
        form.addRow(tr("Column") + ":", self.anchor)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText(tr("OK"))
        buttons.button(QDialogButtonBox.Cancel).setText(tr("Cancel"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def names(self) -> list[str]:
        return [i.text() for i in map(self.columns.item, range(self.columns.count())) if i.checkState() == Qt.Checked]


class ColumnPropertiesDialog(QDialog):
    """Edits a column's description and value order."""

    def __init__(self, ws: Worksheet, name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{tr('Column Properties')}: {name}")
        self.description = QLineEdit(ws.descriptions.get(name, ""))
        self.order = QComboBox()
        for key, label in (("alphabetical", "Alphabetical"), ("occurrence", "Order of occurrence"),
                           ("user", "User-defined")):
            self.order.addItem(tr(label), key)
        current = ws.value_orders.get(name)
        self.order.setCurrentIndex(2 if isinstance(current, list) else 1 if current == "occurrence" else 0)
        self.levels_list = QListWidget()
        self.levels_list.addItems(ws.levels(name))
        up, down = QPushButton(tr("Move Up")), QPushButton(tr("Move Down"))
        up.clicked.connect(lambda: self._move(-1))
        down.clicked.connect(lambda: self._move(1))
        self._buttons = (up, down)
        self.order.setEnabled(not ws.is_numeric(name))
        self.order.currentIndexChanged.connect(self._sync)
        self._sync()
        side = QVBoxLayout()
        side.addWidget(up)
        side.addWidget(down)
        side.addStretch()
        row = QHBoxLayout()
        row.addWidget(self.levels_list)
        row.addLayout(side)
        form = QFormLayout(self)
        form.addRow(tr("Description") + ":", self.description)
        form.addRow(tr("Value Order") + ":", self.order)
        form.addRow(row)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText(tr("OK"))
        buttons.button(QDialogButtonBox.Cancel).setText(tr("Cancel"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _sync(self) -> None:
        user = self.order.isEnabled() and self.order.currentData() == "user"
        self.levels_list.setEnabled(user)
        for button in self._buttons:
            button.setEnabled(user)

    def _move(self, step: int) -> None:
        row = self.levels_list.currentRow()
        if row >= 0 and 0 <= row + step < self.levels_list.count():
            self.levels_list.insertItem(row + step, self.levels_list.takeItem(row))
            self.levels_list.setCurrentRow(row + step)

    def values(self) -> tuple[str, str | list[str] | None]:
        kind = self.order.currentData() if self.order.isEnabled() else "alphabetical"
        if kind == "user":
            return self.description.text(), [self.levels_list.item(i).text() for i in range(self.levels_list.count())]
        return self.description.text(), "occurrence" if kind == "occurrence" else None


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
            lines = [self.ws.descriptions[name]] if name in self.ws.descriptions else []
            if name in self.ws.formulas:
                error = self.ws.formula_errors.get(name)
                lines.append(f"= {self.ws.formulas[name]}" + (f"\n{tr(error)}" if error else ""))
            return "\n".join(lines) or None
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
        if not self.selectionModel().isColumnSelected(section):
            self.selectColumn(section)
        menu.addAction(tr("Rename column") + "...", lambda: self.rename_column(section))
        menu.addAction(tr("Insert Columns"), self.insert_columns)
        menu.addAction(tr("Delete Columns"), self.delete_selection)
        menu.addAction(tr("Move Columns") + "...", self.move_columns)
        menu.addAction(tr("Column Properties") + "...", lambda: self.column_properties(section))
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

    def column_properties(self, section: int) -> None:
        if not 0 <= section < self.ws.ncols:
            return
        name = self.ws.df.columns[section]
        dialog = ColumnPropertiesDialog(self.ws, name, self)
        if dialog.exec() != QDialog.Accepted:
            return
        description, order = dialog.values()

        def apply():
            for store, value in ((self.ws.descriptions, description), (self.ws.value_orders, order)):
                if value:
                    store[name] = value
                else:
                    store.pop(name, None)
        self._apply("Column Properties", apply)

    def keyPressEvent(self, event):
        editing = self.state() == QTableView.EditingState
        if event.matches(QKeySequence.Copy):
            self.copy_selection()
        elif event.matches(QKeySequence.Cut):
            self.cut()
        elif event.matches(QKeySequence.Paste):
            self.paste()
        elif event.key() == Qt.Key_Delete and not editing:
            self.delete_selection()
        elif event.key() == Qt.Key_Backspace and not editing:
            self.clear_selection()
        else:
            super().keyPressEvent(event)

    def _cells(self) -> list[tuple[int, int]]:
        return [(i.row(), i.column()) for i in self.selectedIndexes()]

    def _apply(self, text: str, action: Callable[[], object]) -> None:
        self.change(text, action)
        self.refresh()

    def cut(self) -> None:
        def apply():
            self.copy_selection()
            self.delete_selection()
        self._apply("Cut", apply)

    def delete_selection(self) -> None:
        cells = self._cells()
        rows, cols = sorted({r for r, _ in cells}), sorted({c for _, c in cells})
        chosen = self.selectionModel()
        if cols and all(chosen.isColumnSelected(c) for c in cols):
            self._apply("Delete Cells", lambda: self.ws.delete_columns(cols))
        elif rows and all(chosen.isRowSelected(r) for r in rows):
            self._apply("Delete Cells", lambda: self.ws.delete_rows(rows))
        else:
            self._apply("Delete Cells", lambda: self.ws.delete_cells(cells))

    def clear_selection(self) -> None:
        cells = self._cells()
        self._apply("Clear Cells", lambda: self.ws.clear_cells(cells))

    def insert_cells(self) -> None:
        cells = self._cells()
        self._apply("Insert Cells", lambda: self.ws.insert_cells(cells))

    def insert_rows(self) -> None:
        rows = sorted({r for r, _ in self._cells()})
        if rows:
            self._apply("Insert Rows", lambda: self.ws.insert_rows(rows[0], len(rows)))

    def insert_columns(self) -> None:
        cols = sorted({c for _, c in self._cells()})
        if cols:
            self._apply("Insert Columns", lambda: self.ws.insert_columns(cols[0], len(cols)))

    def move_columns(self) -> None:
        names = list(self.ws.df.columns)
        picked = {c for _, c in self._cells() if c < len(names)}
        dialog = MoveColumnsDialog(names, [names[c] for c in picked], self)
        if dialog.exec() != QDialog.Accepted or not dialog.names():
            return
        try:
            self._apply("Move Columns", lambda: self.ws.move_columns(
                dialog.names(), dialog.where.currentData(), dialog.anchor.currentText()))
        except ValueError as exc:
            QMessageBox.warning(self, tr("Error"), tr(str(exc)))
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
