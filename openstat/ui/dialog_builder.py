"""Builds an analysis dialog from its parameter declarations."""
from __future__ import annotations

from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QLineEdit, QListWidget, QMessageBox, QSpinBox)

from openstat.core.registry import Analysis, Param
from openstat.core.worksheet import Worksheet
from openstat.i18n import tr


def candidate_columns(ws: Worksheet, param: Param) -> list[str]:
    return ws.numeric_columns() if param.numeric else ws.column_names()


class AnalysisDialog(QDialog):
    def __init__(self, analysis: Analysis, ws: Worksheet, parent=None):
        super().__init__(parent)
        self.analysis = analysis
        self.ws = ws
        self.widgets: dict[str, object] = {}
        self.setWindowTitle(tr(analysis.title))
        form = QFormLayout(self)
        for param in analysis.params:
            widget = self._make_widget(param)
            self.widgets[param.name] = widget
            if param.kind == "bool":
                form.addRow(widget)
            else:
                form.addRow(tr(param.label) + ":", widget)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText(tr("OK"))
        buttons.button(QDialogButtonBox.Cancel).setText(tr("Cancel"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _make_widget(self, param: Param):
        if param.kind == "columns":
            w = QListWidget()
            w.setSelectionMode(QAbstractItemView.ExtendedSelection)
            w.addItems(candidate_columns(self.ws, param))
            w.setMinimumHeight(120)
            return w
        if param.kind == "column":
            w = QComboBox()
            if param.optional:
                w.addItem(tr("(none)"), None)
            for name in candidate_columns(self.ws, param):
                w.addItem(name, name)
            return w
        if param.kind == "number":
            w = QDoubleSpinBox()
            w.setDecimals(6)
            w.setRange(param.minimum if param.minimum is not None else -1e12,
                       param.maximum if param.maximum is not None else 1e12)
            w.setValue(float(param.default or 0.0))
            return w
        if param.kind == "int":
            w = QSpinBox()
            w.setRange(int(param.minimum if param.minimum is not None else -10**9),
                       int(param.maximum if param.maximum is not None else 10**9))
            w.setValue(int(param.default or 0))
            return w
        if param.kind == "choice":
            w = QComboBox()
            for choice in param.choices:
                w.addItem(tr(choice), choice)
            if param.default in param.choices:
                w.setCurrentIndex(param.choices.index(param.default))
            return w
        if param.kind == "bool":
            w = QCheckBox(tr(param.label))
            w.setChecked(bool(param.default))
            return w
        if param.kind == "text":
            return QLineEdit(str(param.default or ""))
        raise ValueError(f"Unknown parameter kind: {param.kind}")

    def values(self) -> dict:
        out = {}
        for param in self.analysis.params:
            w = self.widgets[param.name]
            if param.kind == "columns":
                out[param.name] = [item.text() for item in w.selectedItems()]
            elif param.kind in ("column", "choice"):
                out[param.name] = w.currentData()
            elif param.kind in ("number", "int"):
                out[param.name] = w.value()
            elif param.kind == "bool":
                out[param.name] = w.isChecked()
            else:
                out[param.name] = w.text()
        return out

    def accept(self):
        values = self.values()
        for param in self.analysis.params:
            if param.optional:
                continue
            if param.kind == "columns" and not values[param.name]:
                QMessageBox.warning(self, tr("Error"), f"{tr('Select at least one column for')} {tr(param.label)}.")
                return
            if param.kind == "column" and values[param.name] is None:
                QMessageBox.warning(self, tr("Error"), f"{tr('Select a column for')} {tr(param.label)}.")
                return
        super().accept()
