"""Main window: navigator, output pane, worksheet tabs and command line."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QUrl
from PySide6.QtGui import (QAction, QColor, QDesktopServices, QIcon, QKeySequence, QPainter, QPdfWriter, QPixmap,
                           QUndoGroup)
from PySide6.QtWidgets import (QAbstractItemView, QDialog, QDialogButtonBox, QDockWidget, QFileDialog, QFormLayout,
                               QInputDialog, QLineEdit, QListWidget, QMainWindow, QMenu, QMessageBox, QPlainTextEdit,
                               QSplitter, QStyle, QTabWidget, QToolBar)

from openstat import __version__
from openstat.core import io, registry
from openstat.core.export import export_docx
from openstat.core.project import OutputItem, Project
from openstat.core.result import Result, wrap_document
from openstat.core.worksheet import Worksheet
from openstat.i18n import language, tr
from openstat.ui.assistant_dialog import AssistantDialog, GraphBuilderDialog
from openstat.ui.console import Console
from openstat.ui.dialog_builder import AnalysisDialog
from openstat.ui.output_view import OutputView
from openstat.ui.worksheet_view import WorksheetView

PROJECT_FILTER = "OpenStat project (*.ostat)"
MENU_ORDER = ["Data", "Calc", "Stat", "Graph"]
STAT_ORDER = ["Basic Statistics", "Regression", "ANOVA", "DOE", "Control Charts", "Quality Tools",
              "Reliability/Survival", "Predictive Analytics", "Multivariate", "Time Series", "Tables",
              "Nonparametrics", "Equivalence Tests", "Power and Sample Size"]
LANGUAGES = {"pt_BR": "Português (Brasil)", "en": "English"}


SCRIPT_FILTER = "Python (*.py)"


class QueryDialog(QDialog):
    """Asks where the data are and which query brings them."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Import from Database..."))
        self.setMinimumWidth(520)
        self.connection, self.query = QLineEdit(), QPlainTextEdit("SELECT * FROM ")
        self.connection.setPlaceholderText(tr("SQLite file, or an ODBC connection string such as DSN=name"))
        form = QFormLayout(self)
        form.addRow(tr("Connection") + ":", self.connection)
        form.addRow(tr("Query") + ":", self.query)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText(tr("OK"))
        buttons.button(QDialogButtonBox.Cancel).setText(tr("Cancel"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)


def app_icon() -> QIcon:
    """Icon drawn in code: three bars and a control line."""
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor("#1f4e79"))
    painter.drawRoundedRect(0, 0, 64, 64, 10, 10)
    painter.setBrush(QColor("#ffffff"))
    for x, height in ((12, 22), (27, 38), (42, 30)):
        painter.drawRect(x, 54 - height, 10, height)
    painter.setPen(QColor("#ffc000"))
    painter.drawLine(8, 14, 56, 14)
    painter.end()
    return QIcon(pixmap)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.project = Project()
        self.dirty = False
        self.session: list[str] = []   # the analyses run so far, as console commands
        self.resize(1200, 800)
        self.setWindowIcon(app_icon())

        self.output = OutputView()
        self.tabs = QTabWidget()
        self.tabs.tabBarDoubleClicked.connect(self.rename_worksheet)
        self.undo_group = QUndoGroup(self)
        self.tabs.currentChanged.connect(
            lambda i: self.undo_group.setActiveStack(self.tabs.widget(i).undo_stack if i >= 0 else None))
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.output)
        splitter.addWidget(self.tabs)
        splitter.setSizes([400, 400])
        self.setCentralWidget(splitter)

        self.navigator = QListWidget()
        self.navigator.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.navigator.currentRowChanged.connect(self._navigate)
        dock = QDockWidget(tr("Navigator"), self)
        dock.setWidget(self.navigator)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        self.console = Console(self)
        self.console_dock = QDockWidget(tr("Command Line"), self)
        self.console_dock.setWidget(self.console)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.console_dock)
        self.console_dock.hide()

        self._build_menus()
        self.set_project(Project())

    # ----- menus -----

    def _action(self, menu: QMenu, text: str, slot, shortcut=None, translate=True) -> QAction:
        action = menu.addAction(tr(text) if translate else text)
        action.triggered.connect(lambda _checked=False: slot())
        if shortcut:
            action.setShortcut(shortcut)
        return action

    def _build_menus(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu(tr("File"))
        self._action(file_menu, "New Project", self.new_project, QKeySequence.New)
        self._action(file_menu, "New Worksheet", self.new_worksheet)
        open_action = self._action(file_menu, "Open Project...", self.open_project, QKeySequence.Open)
        save_action = self._action(file_menu, "Save Project", self.save_project, QKeySequence.Save)
        self._action(file_menu, "Save Project As...", self.save_project_as)
        file_menu.addSeparator()
        self._action(file_menu, "Import Data...", self.import_data)
        self._action(file_menu, "Import from Database...", self.import_database)
        self._action(file_menu, "Export Worksheet as CSV...", self.export_worksheet)
        file_menu.addSeparator()
        for title, selected in (("Export Output", False), ("Export Selected Output", True)):
            export_menu = file_menu.addMenu(tr(title))
            for label, kind in (("HTML...", "html"), ("PDF...", "pdf"), ("Word (.docx)...", "docx")):
                self._action(export_menu, label, lambda kind=kind, selected=selected:
                             self.export_output(kind, selected=selected), translate=False)
        self._action(file_menu, "Add Note to Output...", self.add_note)
        self._action(file_menu, "Remove Selected Output", self.remove_selected_output)
        self._action(file_menu, "Clear Output", self.clear_output)
        file_menu.addSeparator()
        self._action(file_menu, "Exit", self.close)

        edit_menu = bar.addMenu(tr("Edit"))
        undo = self.undo_group.createUndoAction(self, tr("Undo"))
        undo.setShortcut(QKeySequence.Undo)
        redo = self.undo_group.createRedoAction(self, tr("Redo"))
        redo.setShortcut("Ctrl+Y")
        edit_menu.addActions([undo, redo])
        edit_menu.addSeparator()
        # grid shortcuts live on the tabs, so they do not fire while another pane has the focus
        def grid(text, method, shortcut=None):
            action = self._action(edit_menu, text, lambda: getattr(self.current_view(), method)(), shortcut)
            action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            self.tabs.addAction(action)
            return action
        cut, copy, paste = grid("Cut", "cut", QKeySequence.Cut), grid("Copy", "copy_selection", QKeySequence.Copy),             grid("Paste", "paste", QKeySequence.Paste)
        edit_menu.addSeparator()
        grid("Delete Cells", "delete_selection", QKeySequence.Delete)
        grid("Clear Cells", "clear_selection")
        grid("Select All", "selectAll", QKeySequence.SelectAll)
        edit_menu.addSeparator()
        inserts = [grid("Insert Cells", "insert_cells"), grid("Insert Rows", "insert_rows"),
                   grid("Insert Columns", "insert_columns")]
        move = grid("Move Columns", "move_columns")
        move.setText(move.text() + "...")

        toolbar = self.addToolBar(tr("Standard"))
        toolbar.setObjectName("Standard")
        style = self.style()
        icons = [(open_action, QStyle.SP_DialogOpenButton), (save_action, QStyle.SP_DialogSaveButton), None,
                 (cut, QStyle.SP_FileIcon), (copy, QStyle.SP_FileLinkIcon), (paste, QStyle.SP_DialogApplyButton), None,
                 (undo, QStyle.SP_ArrowBack), (redo, QStyle.SP_ArrowForward), None,
                 (inserts[0], QStyle.SP_FileDialogNewFolder), (inserts[1], QStyle.SP_ArrowDown),
                 (inserts[2], QStyle.SP_ArrowRight), (move, QStyle.SP_FileDialogDetailedView)]
        for entry in icons:
            if entry is None:
                toolbar.addSeparator()
                continue
            action, fallback = entry
            action.setIcon(style.standardIcon(fallback))
            toolbar.addAction(action)

        assistant_menu = bar.addMenu(tr("Assistant"))
        self._action(assistant_menu, "Choose an Analysis...", self.open_assistant)
        self._action(assistant_menu, "Graph Builder...", self.open_graph_builder)

        menus: dict[tuple, QMenu] = {}
        def submenu_rank(a):
            name = a.menu[1] if len(a.menu) > 1 else ""
            return STAT_ORDER.index(name) if name in STAT_ORDER else len(STAT_ORDER)

        analyses = sorted(registry.all_analyses(), key=submenu_rank)   # stable: keeps registration order
        tops = MENU_ORDER + sorted({a.menu[0] for a in analyses} - set(MENU_ORDER))
        for top in tops:
            if any(a.menu[0] == top for a in analyses):
                menus[(top,)] = bar.addMenu(tr(top))
        for a in analyses:
            for depth in range(2, len(a.menu) + 1):
                path = a.menu[:depth]
                if path not in menus:
                    menus[path] = menus[path[:-1]].addMenu(tr(path[-1]))
            self._action(menus[a.menu], tr(a.title) + "...", lambda a=a: self.run_analysis(a), translate=False)

        tools_menu = bar.addMenu(tr("Tools"))
        toggle = self.console_dock.toggleViewAction()
        toggle.setText(tr("Command Line"))
        toggle.setShortcut("Ctrl+L")
        tools_menu.addAction(toggle)
        self._action(tools_menu, "Save Session as Script...", self.save_session)
        self._action(tools_menu, "Run Script...", self.run_script)
        self._action(tools_menu, "Open Plugins Folder", self.open_plugins_folder)
        language_menu = tools_menu.addMenu(tr("Language"))
        for code_, name in LANGUAGES.items():
            action = self._action(language_menu, name, lambda c=code_: self.set_language(c), translate=False)
            action.setCheckable(True)
            action.setChecked(code_ == language())

        help_menu = bar.addMenu(tr("Help"))
        self._action(help_menu, "About OpenStat", self.about)

    # ----- project -----

    def set_project(self, project: Project) -> None:
        self.project = project
        if not project.worksheets:
            project.new_worksheet()
        self.tabs.clear()
        for stack in self.undo_group.stacks():   # the tabs are made anew, with empty histories
            self.undo_group.removeStack(stack)
        for ws in project.worksheets:
            self._add_tab(ws)
        self._refresh_output()
        self.dirty = False
        self._update_title()

    def _add_tab(self, ws: Worksheet) -> None:
        view = WorksheetView(ws)
        view.model_.modelReset.connect(self.mark_dirty)
        view.undo_stack.indexChanged.connect(self.mark_dirty)
        self.undo_group.addStack(view.undo_stack)
        self.tabs.addTab(view, ws.name)

    def mark_dirty(self) -> None:
        if not self.dirty:
            self.dirty = True
            self._update_title()

    def _update_title(self) -> None:
        name = self.project.path.name if self.project.path else tr("New Project")
        self.setWindowTitle(f"{name}{' *' if self.dirty else ''} - OpenStat")

    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        answer = QMessageBox.question(self, "OpenStat", tr("Discard the current project?"))
        return answer == QMessageBox.Yes

    def closeEvent(self, event):
        if self._confirm_discard():
            event.accept()
        else:
            event.ignore()

    def new_project(self) -> None:
        if self._confirm_discard():
            self.set_project(Project())

    def new_worksheet(self) -> None:
        self.add_worksheet(self.project.new_worksheet(), register=False)

    def add_worksheet(self, ws: Worksheet, register: bool = True) -> None:
        if register:
            self.project.worksheets.append(ws)
        self._add_tab(ws)
        self.tabs.setCurrentIndex(self.tabs.count() - 1)
        self.mark_dirty()

    def rename_worksheet(self, index: int) -> None:
        if index < 0:
            return
        ws = self.tabs.widget(index).ws
        name, ok = QInputDialog.getText(self, tr("Rename worksheet"), tr("Worksheet name:"), text=ws.name)
        if ok and name.strip():
            ws.name = name.strip()
            self.tabs.setTabText(index, ws.name)
            self.mark_dirty()

    def worksheets_changed(self) -> None:
        """Refresh every grid after data were changed outside the grid itself."""
        for i in range(self.tabs.count()):
            self.tabs.widget(i).refresh()

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
            self._save(self.project.path)

    def save_project_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, tr("Save Project As..."), "", tr(PROJECT_FILTER))
        if path:
            self._save(path if path.lower().endswith(".ostat") else path + ".ostat")

    def _save(self, path) -> None:
        def save():
            self.project.save(path)
            self.dirty = False
        self._guarded(save)
        self._update_title()

    def import_data(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr("Import Data..."), "", io.IMPORT_FILTER)
        if path:
            self._guarded(lambda: [self.add_worksheet(ws) for ws in io.import_file(path)])

    def import_database(self) -> None:
        dialog = QueryDialog(self)
        if dialog.exec() == QDialog.Accepted:
            self._guarded(lambda: self.add_worksheet(io.import_sql(dialog.connection.text(),
                                                                   dialog.query.toPlainText())))

    def export_worksheet(self) -> None:
        ws = self.current_worksheet()
        path, _ = QFileDialog.getSaveFileName(self, tr("Export Worksheet as CSV..."), f"{ws.name}.csv", "CSV (*.csv)")
        if path:
            self._guarded(lambda: io.export_csv(ws, path))

    def selected_outputs(self) -> list[int]:
        """Positions of the output items selected in the navigator."""
        return sorted(index.row() for index in self.navigator.selectedIndexes())

    def output_html(self, selected: bool = False) -> str:
        outputs = [self.project.outputs[i] for i in self.selected_outputs()] if selected else self.project.outputs
        return wrap_document("\n<hr>\n".join(o.html for o in outputs))

    def export_output(self, kind: str, path: str | None = None, selected: bool = False) -> None:
        if selected and not self.selected_outputs():
            QMessageBox.information(self, "OpenStat", tr("Select the items to export in the navigator."))
            return
        filters = {"html": "HTML (*.html)", "pdf": "PDF (*.pdf)", "docx": "Word (*.docx)"}
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, tr("Export Output"), f"output.{kind}", filters[kind])
        if not path:
            return
        if kind == "html":
            self._guarded(lambda: Path(path).write_text(self.output_html(selected), encoding="utf-8"))
        elif kind == "docx":
            self._guarded(lambda: export_docx(self.output_html(selected), path))
        else:
            view = self.output
            if selected:   # a view of its own lays out just the chosen items, with their figures
                view = OutputView()
                view.show_outputs([self.project.outputs[i] for i in self.selected_outputs()])
            self._guarded(lambda: view.document().print_(QPdfWriter(str(path))))

    def add_note(self, text: str | None = None) -> None:
        if text is None:
            text, ok = QInputDialog.getMultiLineText(self, tr("Add Note to Output..."), tr("Note") + ":")
            if not ok:
                return
        if text.strip():
            self.add_result(Result("Note").text(text.strip()), self.current_worksheet())

    def remove_selected_output(self) -> None:
        chosen = set(self.selected_outputs())
        if chosen:
            self.project.outputs = [o for i, o in enumerate(self.project.outputs) if i not in chosen]
            self._refresh_output()
            self.mark_dirty()

    def clear_output(self) -> None:
        self.project.outputs.clear()
        self._refresh_output()
        self.mark_dirty()

    def _guarded(self, action) -> None:
        try:
            action()
        except Exception as exc:  # shown to the user instead of crashing the app
            QMessageBox.critical(self, tr("Error"), tr(str(exc)) if isinstance(exc, ValueError)
                                 else f"{type(exc).__name__}: {exc}")

    # ----- tools -----

    def open_plugins_folder(self) -> None:
        folder = registry.plugins_dir()
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def record(self, a: registry.Analysis, values: dict) -> None:
        """Remember an analysis that was run, as the console command that repeats it."""
        self.session.append(registry.format_call(a, values))

    def session_script(self) -> str:
        return "# OpenStat session: run with Tools > Run Script, on a worksheet with the same columns\n" \
            + "\n".join(self.session) + "\n"

    def save_session(self, path: str | None = None) -> None:
        if not self.session:
            QMessageBox.information(self, "OpenStat", tr("No analysis has been run in this session yet."))
            return
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, tr("Save Session as Script..."), "session.py", SCRIPT_FILTER)
        if path:
            self._guarded(lambda: Path(path).write_text(self.session_script(), encoding="utf-8"))

    def run_script(self, path: str | None = None) -> None:
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, tr("Run Script..."), "", SCRIPT_FILTER)
        if not path:
            return
        self.console_dock.show()   # what the script prints, and its errors, appear in the command line
        self._guarded(lambda: self.console.run_script(Path(path).read_text(encoding="utf-8"), Path(path).name))

    def open_assistant(self) -> None:
        dialog = AssistantDialog(self)
        if dialog.exec() == QDialog.Accepted and dialog.chosen:
            self.run_analysis(registry.get(dialog.chosen))

    def open_graph_builder(self) -> None:
        ws = self.current_worksheet()
        if ws.ncols == 0:
            QMessageBox.information(self, "OpenStat", tr("The worksheet has no data."))
            return
        dialog = GraphBuilderDialog(ws, self)
        if dialog.exec() == QDialog.Accepted and dialog.chosen:
            self.run_analysis(*dialog.chosen)

    def set_language(self, code_: str) -> None:
        QSettings("OpenStat", "OpenStat").setValue("language", code_)
        QMessageBox.information(self, "OpenStat", tr("The language will change the next time OpenStat starts."))

    # ----- analyses -----

    def current_view(self) -> WorksheetView:
        return self.tabs.currentWidget()

    def current_worksheet(self) -> Worksheet:
        return self.current_view().ws

    def view_of(self, ws: Worksheet) -> WorksheetView:
        return next(view for view in map(self.tabs.widget, range(self.tabs.count())) if view.ws is ws)

    def run_analysis(self, a: registry.Analysis, initial: dict | None = None) -> None:
        ws = self.current_worksheet()
        if ws.ncols == 0 and any(p.kind in ("column", "columns") for p in a.params):
            QMessageBox.information(self, "OpenStat", tr("The worksheet has no data."))
            return
        dialog = AnalysisDialog(a, ws, self, self.project.worksheets, initial)
        if dialog.exec() != AnalysisDialog.Accepted:
            return

        def run():
            values = dialog.values()
            self.add_result(a.func(ws, **values), ws)
            self.record(a, values)
        self._guarded(run)

    def add_result(self, result: Result, ws: Worksheet) -> None:
        def store():
            for name, values in result.stored.items():
                ws.add_column(name, values)
            for name, expression in result.formulas.items():
                ws.set_formula(name, expression)
        if result.stored or result.formulas:
            self.view_of(ws).change(result.title, store)
            self.worksheets_changed()
        for name, df in result.worksheets:
            self.add_worksheet(Worksheet(name, df))
        index = len(self.project.outputs)
        self.project.outputs.append(OutputItem(tr(result.title), result.to_html(anchor=f"r{index}")))
        self._refresh_output(scroll_to=index)
        self.mark_dirty()

    def _refresh_output(self, scroll_to: int | None = None) -> None:
        self.navigator.blockSignals(True)
        self.navigator.clear()
        self.navigator.addItems([o.title for o in self.project.outputs])
        self.navigator.blockSignals(False)
        self.output.show_outputs(self.project.outputs, scroll_to)

    def _navigate(self, row: int) -> None:
        if row >= 0:
            self.output.scrollToAnchor(f"o{row}")

    def about(self) -> None:
        QMessageBox.about(self, tr("About OpenStat"),
                          f"OpenStat {__version__}\n\n{tr('Open-source statistical analysis application.')}\n\n"
                          f"{len(registry.all_analyses())} {tr('analyses available')}\n\nMIT License")
