import pandas as pd
import pytest
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QInputDialog

from openstat.core.formatting import MISSING, make_rule
from openstat.core.result import Result
from openstat.core.worksheet import Worksheet, changed_parts


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


def test_typing_in_a_cell_still_opens_the_editor(window):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QTableView

    view = window.current_view(); window.show()
    view.setCurrentIndex(view.model_.index(0, 0))
    QTest.keyClick(view, Qt.Key_5)
    assert view.state() == QTableView.EditingState


def test_no_step_when_nothing_changed(window):
    window.current_view().model_.setData(window.current_view().model_.index(0, 0), "1")   # same value
    assert window.current_view().undo_stack.count() == 0
