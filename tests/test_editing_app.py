import pandas as pd
import pytest
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QInputDialog, QTableView, QToolBar

from openstat.core.formatting import MISSING, make_rule
from openstat.core.result import Result
from openstat.core.worksheet import Worksheet, changed_parts
from openstat.i18n import tr


@pytest.fixture
def window(qapp):
    from openstat.ui.main_window import MainWindow

    w = MainWindow()
    w.add_worksheet(Worksheet("Data", pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0], "B": [2.0, 4.0, 6.0, 9.0]})))
    return w


def same_state(a: dict, b: dict) -> bool:
    before, after = changed_parts(a, b)
    return not before["data"] and not after["data"] and before == after


def test_typing_beyond_data_undo_restores_size(window):        # Review Focus 1
    view = window.current_view(); ws = view.ws
    view.model_.setData(view.model_.index(9, 5), "1")
    view.undo_stack.undo(); assert (ws.nrows, ws.ncols) == (4, 2)
    view.undo_stack.redo(); assert ws.df.iat[9, 5] == 1.0


def test_each_point_is_one_undo_step(window, monkeypatch):
    view = window.current_view(); ws = view.ws

    def paste():
        QGuiApplication.clipboard().setText("7\t8\n9\t10")
        view.setCurrentIndex(view.model_.index(1, 0))
        view.paste()

    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Renamed", True))
    steps = [paste, lambda: view.rename_column(1),
             lambda: view.set_formats("A", [make_rule(MISSING)]),
             lambda: window.console.execute("df['C'] = df['A']*2")]
    for i, step in enumerate(steps):
        before = ws.snapshot()
        step()
        assert view.undo_stack.count() == i + 1
        view.undo_stack.undo()
        assert same_state(before, ws.snapshot())
        view.undo_stack.redo()


def test_console_in_place_change_is_undone(window):           # Review Focus 5
    window.console.execute("df.loc[0, 'A'] = 99")
    window.current_view().undo_stack.undo()
    assert window.current_worksheet().df["A"][0] == 1.0


def test_undo_analysis_keeps_output_and_new_worksheets(window):
    window.console.execute("run('data.rank', column='A')")      # stores Rank_A
    window.console.execute("run('data.sort', by=['A'])")        # new worksheet, no column change
    view = window.view_of(window.project.worksheets[1]); view.undo_stack.undo()
    assert "Rank_A" not in view.ws.df and len(window.project.outputs) == 2 and len(window.project.worksheets) == 3


def test_failing_result_leaves_no_half_column(window):        # Review Focus 2
    res = Result("X"); res.stored["Good"] = [1, 2, 3, 4]; res.formulas["Bad"] = "Bad + 1"
    with pytest.raises(ValueError): window.add_result(res, window.current_worksheet())
    assert "Good" not in window.current_worksheet().df and window.current_view().undo_stack.count() == 0


def test_undo_history_is_per_worksheet(window):
    window.tabs.setCurrentIndex(1)
    a = window.current_view()
    a.model_.setData(a.model_.index(0, 0), "50")
    window.tabs.setCurrentIndex(0)
    window.undo_group.undo()
    assert a.ws.df.iat[0, 0] == 50.0 and a.undo_stack.count() == 1 and a.undo_stack.canUndo()


def test_typing_in_a_cell_opens_the_editor_and_records_a_step(window):
    from PySide6.QtWidgets import QApplication, QTableView

    view = window.current_view(); window.show()
    view.setCurrentIndex(view.model_.index(0, 0))
    QTest.keyClick(view, Qt.Key_5)
    assert view.state() == QTableView.EditingState
    QTest.keyClick(view.indexWidget(view.currentIndex()), Qt.Key_Return)
    QApplication.processEvents()   # the delegate commits on Return through a queued call
    assert view.ws.df.iat[0, 0] == 5.0 and view.undo_stack.count() == 1


def test_no_step_when_nothing_changed(window):
    window.current_view().model_.setData(window.current_view().model_.index(0, 0), "1")   # same value
    assert window.current_view().undo_stack.count() == 0


def test_delete_key_deletes_and_shifts_up(window):
    view = window.current_view(); view.setCurrentIndex(view.model_.index(0, 0))
    QTest.keyClick(view, Qt.Key_Delete)
    assert view.ws.df["A"].tolist()[:3] == [2.0, 3.0, 4.0] and view.undo_stack.undoText() == tr("Delete Cells")


def test_backspace_clears_in_place(window):
    view = window.current_view(); view.setCurrentIndex(view.model_.index(0, 0))
    QTest.keyClick(view, Qt.Key_Backspace)
    assert pd.isna(view.ws.df["A"][0]) and view.ws.df["A"][1] == 2.0 and view.undo_stack.undoText() == tr("Clear Cells")


def test_whole_column_selected_delete_removes_column(window):
    view = window.current_view(); view.selectColumn(0); view.delete_selection()
    assert view.ws.column_names() == ["B"]


def test_whole_row_selected_delete_removes_row(window):
    view = window.current_view(); view.selectRow(0); view.delete_selection()
    assert view.ws.nrows == 3


def test_cut_is_one_step_and_fills_clipboard(window):
    view = window.current_view(); view.setCurrentIndex(view.model_.index(0, 1)); view.cut()
    assert QGuiApplication.clipboard().text() == "2" and view.undo_stack.count() == 1
    view.undo_stack.undo(); assert view.ws.df["B"][0] == 2.0


def test_insert_rows_and_columns_from_selection(window):
    view = window.current_view()
    view.selectionModel().select(view.model_.index(1, 0), QItemSelectionModel.ClearAndSelect)
    view.selectionModel().select(view.model_.index(2, 0), QItemSelectionModel.Select)
    view.insert_rows(); assert view.ws.nrows == 6
    view.selectColumn(0); view.insert_columns()
    assert view.ws.column_names()[1] == "A" and view.undo_stack.count() == 2


def test_move_columns_dialog(window, monkeypatch):
    from PySide6.QtWidgets import QDialog
    from openstat.ui.worksheet_view import MoveColumnsDialog
    view = window.current_view(); view.selectColumn(0)
    monkeypatch.setattr(MoveColumnsDialog, "exec", lambda self: (self.where.setCurrentIndex(1), QDialog.Accepted)[1])
    view.move_columns()
    assert view.ws.column_names() == ["B", "A"] and view.undo_stack.undoText() == tr("Move Columns")


def test_edit_menu_and_toolbar_exist(window):
    titles = [a.text() for a in window.menuBar().actions()]
    assert titles[1] == tr("Edit") and window.findChild(QToolBar) is not None


def test_shortcuts_fire_once_and_not_while_editing(window):
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    view = window.current_view(); window.show(); view.setFocus(); QApplication.processEvents()
    view.setCurrentIndex(view.model_.index(0, 1))
    QTest.keyClick(view, Qt.Key_X, Qt.ControlModifier)
    assert QGuiApplication.clipboard().text() == "2" and view.undo_stack.count() == 1
    view.setCurrentIndex(view.model_.index(0, 0))
    QTest.keyClick(view, Qt.Key_Delete)
    assert view.undo_stack.count() == 2
    view.setCurrentIndex(view.model_.index(1, 0))   # the reset after a change drops the current cell
    before = view.ws.snapshot()
    QTest.keyClick(view, Qt.Key_F2); assert view.state() == QTableView.EditingState
    QTest.keyClick(view.indexWidget(view.currentIndex()), Qt.Key_Delete)
    QTest.keyClick(view.indexWidget(view.currentIndex()), Qt.Key_Backspace)
    assert view.undo_stack.count() == 2 and same_state(before, view.ws.snapshot())


def test_select_all_delete_clears_everything_without_error(window):
    view = window.current_view(); view.selectAll(); view.delete_selection()
    assert view.ws.ncols == 0 and view.undo_stack.count() == 1
    view.undo_stack.undo(); assert view.ws.column_names() == ["A", "B"] and view.ws.nrows == 4


def test_find_next_wraps_and_respects_options():
    from openstat.ui.find_dialog import find_next

    ws = Worksheet("W", pd.DataFrame({"A": ["ab", "x", "AB"], "B": ["b", "ab", "c"]}))
    assert find_next(ws, "ab", (0, 0), False, False, None) == (1, 1)
    assert find_next(ws, "ab", (1, 1), True, False, None) == (0, 0)          # wraps; "AB" skipped by case
    assert find_next(ws, "a", (0, 0), False, True, None) is None              # whole cell
    assert find_next(ws, "ab", (0, 0), False, False, [0]) == (2, 0)


def test_replace_all_is_one_step(window):
    from openstat.ui.find_dialog import FindDialog

    dialog = FindDialog(window, replace=True); dialog.find_text.setText("2"); dialog.replace_text.setText("5")
    dialog.whole_cell.setChecked(True); dialog.replace_all()
    ws = window.current_worksheet(); assert ws.df["A"][1] == 5.0 and window.current_view().undo_stack.count() == 1


def test_find_replace_actions_and_dialog_reuse(window):
    window.open_find(False); first = window._find_dialog
    window.open_find(False)
    assert window._find_dialog is first
    dialog = window._find_dialog; dialog.find_text.setText("3"); dialog.find_next()
    assert window.current_view().currentIndex().row() == 2
