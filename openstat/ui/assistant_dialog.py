"""Dialogs that help choose what to run: the assistant's questions and the graph builder."""
from __future__ import annotations

from PySide6.QtWidgets import (QAbstractItemView, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QListWidget,
                               QPushButton, QVBoxLayout)

from openstat.core import registry
from openstat.core.assistant import TREE, Node, suggest_graphs
from openstat.core.worksheet import Worksheet
from openstat.i18n import tr


class AssistantDialog(QDialog):
    """Walks the decision tree; `chosen` is the id of the analysis reached, once accepted."""

    def __init__(self, parent=None, tree: Node = TREE):
        super().__init__(parent)
        self.setWindowTitle(tr("Assistant"))
        self.setMinimumWidth(460)
        self.path: list[Node] = [tree]
        self.chosen: str | None = None
        self.question = QLabel(wordWrap=True)
        self.question.setStyleSheet("font-weight: bold; font-size: 11pt;")
        self.options = QVBoxLayout()
        self.back = QPushButton(tr("Back"))
        self.back.clicked.connect(self.go_back)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Cancel).setText(tr("Cancel"))
        buttons.rejected.connect(self.reject)
        bottom = QHBoxLayout()
        bottom.addWidget(self.back)
        bottom.addStretch()
        bottom.addWidget(buttons)
        layout = QVBoxLayout(self)
        layout.addWidget(self.question)
        layout.addLayout(self.options)
        layout.addStretch()
        layout.addLayout(bottom)
        self._show()

    def _show(self) -> None:
        while self.options.count():
            self.options.takeAt(0).widget().deleteLater()
        node = self.path[-1]
        self.back.setEnabled(len(self.path) > 1)
        if node.analysis:
            a = registry.get(node.analysis)
            self.question.setText(tr(a.title))
            advice = QLabel(tr(node.text), wordWrap=True)
            self.options.addWidget(advice)
            run = QPushButton(f"{tr('Open')} {tr(a.title)}...")
            run.clicked.connect(self._accept_leaf)
            self.options.addWidget(run)
            return
        self.question.setText(tr(node.text))
        for label, child in node.options:
            button = QPushButton(tr(label))
            button.setStyleSheet("text-align: left; padding: 6px;")
            button.clicked.connect(lambda _checked=False, child=child: self.choose(child))
            self.options.addWidget(button)

    def choose(self, child: Node) -> None:
        self.path.append(child)
        self._show()

    def go_back(self) -> None:
        if len(self.path) > 1:
            self.path.pop()
            self._show()

    def _accept_leaf(self) -> None:
        self.chosen = self.path[-1].analysis
        self.accept()


class GraphBuilderDialog(QDialog):
    """Pick columns and see the graphs that can be drawn from them; `chosen` is (analysis, values)."""

    def __init__(self, ws: Worksheet, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Graph Builder"))
        self.ws = ws
        self.chosen = None
        self.suggestions: list = []
        self.columns = QListWidget()
        self.columns.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.columns.addItems(ws.column_names())
        self.columns.itemSelectionChanged.connect(self.refresh)
        self.graphs = QListWidget()
        self.graphs.itemDoubleClicked.connect(lambda _item: self.accept())
        lists = QHBoxLayout()
        for title, widget in ((tr("Columns (in the order you select them)"), self.columns),
                              (tr("Graphs for these columns"), self.graphs)):
            side = QVBoxLayout()
            side.addWidget(QLabel(title))
            side.addWidget(widget)
            lists.addLayout(side)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText(tr("OK"))
        buttons.button(QDialogButtonBox.Cancel).setText(tr("Cancel"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(lists)
        layout.addWidget(buttons)
        self.order: list[str] = []

    def selected_columns(self) -> list[str]:
        """The selected columns in the order they were selected: the first is Y, the second X."""
        now = [item.text() for item in self.columns.selectedItems()]
        self.order = [c for c in self.order if c in now] + [c for c in now if c not in self.order]
        return list(self.order)

    def refresh(self) -> None:
        self.suggestions = suggest_graphs(self.ws, self.selected_columns(), registry.all_analyses())
        self.graphs.clear()
        self.graphs.addItems([tr(a.title) for a, _ in self.suggestions])
        if self.suggestions:
            self.graphs.setCurrentRow(0)

    def accept(self):
        row = self.graphs.currentRow()
        if 0 <= row < len(self.suggestions):
            self.chosen = self.suggestions[row]
            super().accept()
