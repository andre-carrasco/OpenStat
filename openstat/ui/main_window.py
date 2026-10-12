"""Main window: navigator, output pane and worksheet tabs."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QDockWidget, QFileDialog, QListWidget, QMainWindow, QMenu, QMessageBox,
                               QSplitter, QTabWidget)

from openstat import __version__
from openstat.core import io, registry
from openstat.core.project import OutputItem, Project
from openstat.core.result import Result, wrap_document
from openstat.core.worksheet import Worksheet
from openstat.i18n import tr
from openstat.ui.dialog_builder import AnalysisDialog
from openstat.ui.output_view import OutputView
from openstat.ui.worksheet_view import WorksheetView

PROJECT_FILTER = "OpenStat project (*.ostat)"
MENU_ORDER = ["Data", "Calc", "Stat", "Graph"]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.project = Project()
        self.resize(1200, 800)

        self.output = OutputView()
        self.tabs = QTabWidget()
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.output)
        splitter.addWidget(self.tabs)
        splitter.setSizes([400, 400])
        self.setCentralWidget(splitter)

        self.navigator = QListWidget()
        self.navigator.currentRowChanged.connect(self._navigate)
        dock = QDockWidget(tr("Navigator"), self)
        dock.setWidget(self.navigator)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        self._build_menus()
        self.set_project(Project())

    # ----- menus -----

    def _action(self, menu: QMenu, text: str, slot, shortcut=None) -> QAction:
        action = menu.addAction(tr(text))
        action.triggered.connect(lambda _checked=False: slot())
        if shortcut:
            action.setShortcut(shortcut)
        return action

    def _build_menus(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu(tr("File"))
        self._action(file_menu, "New Project", self.new_project, QKeySequence.New)
        self._action(file_menu, "New Worksheet", self.new_worksheet)
        self._action(file_menu, "Open Project...", self.open_project, QKeySequence.Open)
        self._action(file_menu, "Save Project", self.save_project, QKeySequence.Save)
        self._action(file_menu, "Save Project As...", self.save_project_as)
        file_menu.addSeparator()
        self._action(file_menu, "Import Data...", self.import_data)
        self._action(file_menu, "Export Worksheet as CSV...", self.export_worksheet)
        self._action(file_menu, "Export Output as HTML...", self.export_output)
        self._action(file_menu, "Clear Output", self.clear_output)
        file_menu.addSeparator()
        self._action(file_menu, "Exit", self.close)

        menus: dict[tuple, QMenu] = {}
        analyses = registry.all_analyses()
        tops = MENU_ORDER + sorted({a.menu[0] for a in analyses} - set(MENU_ORDER))
        for top in tops:
            if any(a.menu[0] == top for a in analyses):
                menus[(top,)] = bar.addMenu(tr(top))
        for a in analyses:
            for depth in range(2, len(a.menu) + 1):
                path = a.menu[:depth]
                if path not in menus:
                    menus[path] = menus[path[:-1]].addMenu(tr(path[-1]))
            self._action(menus[a.menu], a.title + "...", lambda a=a: self.run_analysis(a))

        help_menu = bar.addMenu(tr("Help"))
        self._action(help_menu, "About OpenStat", self.about)

    # ----- project -----

    def set_project(self, project: Project) -> None:
        self.project = project
        if not project.worksheets:
            project.new_worksheet()
        self.tabs.clear()
        for ws in project.worksheets:
            self.tabs.addTab(WorksheetView(ws), ws.name)
        self._refresh_output()
        self._update_title()

    def _update_title(self) -> None:
        name = self.project.path.name if self.project.path else tr("New Project")
        self.setWindowTitle(f"{name} - OpenStat")

    def _has_content(self) -> bool:
        return bool(self.project.outputs) or any(ws.ncols for ws in self.project.worksheets)

    def _confirm_discard(self) -> bool:
        if not self._has_content():
            return True
        answer = QMessageBox.question(self, "OpenStat", tr("Discard the current project?"))
        return answer == QMessageBox.Yes

    def new_project(self) -> None:
        if self._confirm_discard():
            self.set_project(Project())

    def new_worksheet(self) -> None:
        self.add_worksheet(self.project.new_worksheet(), register=False)

    def add_worksheet(self, ws: Worksheet, register: bool = True) -> None:
        if register:
            self.project.worksheets.append(ws)
        self.tabs.addTab(WorksheetView(ws), ws.name)
        self.tabs.setCurrentIndex(self.tabs.count() - 1)

    def open_project(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, tr("Open Project..."), "", tr(PROJECT_FILTER))
        if path:
            self._guarded(lambda: self.set_project(Project.load(path)))

    def save_project(self) -> None:
        if self.project.path is None:
            self.save_project_as()
        else:
            self._guarded(lambda: self.project.save(self.project.path))

    def save_project_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, tr("Save Project As..."), "", tr(PROJECT_FILTER))
        if path:
            if not path.lower().endswith(".ostat"):
                path += ".ostat"
            self._guarded(lambda: self.project.save(path))
            self._update_title()

    def import_data(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr("Import Data..."), "", io.IMPORT_FILTER)
        if path:
            self._guarded(lambda: [self.add_worksheet(ws) for ws in io.import_file(path)])

    def export_worksheet(self) -> None:
        ws = self.current_worksheet()
        path, _ = QFileDialog.getSaveFileName(self, tr("Export Worksheet as CSV..."), f"{ws.name}.csv", "CSV (*.csv)")
        if path:
            self._guarded(lambda: io.export_csv(ws, path))

    def export_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, tr("Export Output as HTML..."), "output.html", "HTML (*.html)")
        if path:
            body = "\n<hr>\n".join(o.html for o in self.project.outputs)
            self._guarded(lambda: Path(path).write_text(wrap_document(body), encoding="utf-8"))

    def clear_output(self) -> None:
        self.project.outputs.clear()
        self._refresh_output()

    def _guarded(self, action) -> None:
        try:
            action()
        except Exception as exc:  # shown to the user instead of crashing the app
            QMessageBox.critical(self, tr("Error"), f"{type(exc).__name__}: {exc}")

    # ----- analyses -----

    def current_view(self) -> WorksheetView:
        return self.tabs.currentWidget()

    def current_worksheet(self) -> Worksheet:
        return self.current_view().ws

    def run_analysis(self, a: registry.Analysis) -> None:
        ws = self.current_worksheet()
        if ws.ncols == 0 and any(p.kind in ("column", "columns") for p in a.params):
            QMessageBox.information(self, "OpenStat", tr("The worksheet has no data."))
            return
        dialog = AnalysisDialog(a, ws, self)
        if dialog.exec() != AnalysisDialog.Accepted:
            return
        self._guarded(lambda: self.add_result(a.func(ws, **dialog.values()), ws))

    def add_result(self, result: Result, ws: Worksheet) -> None:
        for name, values in result.stored.items():
            ws.add_column(name, values)
        if result.stored:
            self.current_view().refresh()
        index = len(self.project.outputs)
        self.project.outputs.append(OutputItem(tr(result.title), result.to_html(anchor=f"r{index}")))
        self._refresh_output(scroll_to=index)

    def _refresh_output(self, scroll_to: int | None = None) -> None:
        self.navigator.blockSignals(True)
        self.navigator.clear()
        self.navigator.addItems([o.title for o in self.project.outputs])
        self.navigator.blockSignals(False)
        self.output.show_outputs(self.project.outputs, scroll_to)

    def _navigate(self, row: int) -> None:
        if row >= 0:
            self.output.scrollToAnchor(f"r{row}")

    def about(self) -> None:
        QMessageBox.about(self, tr("About OpenStat"),
                          f"OpenStat {__version__}\n\n{tr('Open-source statistical analysis application.')}\n\nMIT License")
