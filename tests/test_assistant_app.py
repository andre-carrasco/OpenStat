"""Assistant, graph builder, conditional formatting, notes, database import and session scripts."""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from openstat.core import assistant, formatting, io, registry
from openstat.core.project import Project
from openstat.core.worksheet import Worksheet


@pytest.fixture
def ws():
    return Worksheet("Data", pd.DataFrame({
        "A": [1.0, 2.0, 3.0, 4.0, 50.0, 2.5, 3.5, np.nan], "B": [2.0, 4.0, 6.0, 9.0, 1.0, 5.0, 7.0, 8.0],
        "G": np.array(["x", "y", "x", "y", "x", "y", "x", None], dtype=object)}))


@pytest.fixture
def window(qapp, ws):
    from openstat.ui.main_window import MainWindow

    w = MainWindow()
    w.add_worksheet(ws)
    return w


def test_assistant_tree_leads_to_registered_analyses():
    leaves = list(assistant.leaves())
    assert len(leaves) > 35
    ids = {a.id for a in registry.all_analyses()}
    assert {leaf.analysis for leaf in leaves} <= ids
    assert all(leaf.text and not leaf.options for leaf in leaves)
    assert assistant.TREE.text in assistant.strings() and len(assistant.strings()) > 100

    def depth(node):
        return 1 + max((depth(child) for _, child in node.options), default=0)
    assert depth(assistant.TREE) <= 5                                  # an answer in a few clicks


def test_graph_suggestions_fit_the_columns(ws):
    analyses = registry.all_analyses()

    def suggested(columns):
        return {a.id: values for a, values in assistant.suggest_graphs(ws, columns, analyses)}

    one = suggested(["A"])
    assert {"graph.histogram", "graph.boxplot", "graph.dotplot"} <= set(one) and "graph.scatter" not in one
    assert one["graph.histogram"]["columns"] == ["A"]
    two = suggested(["B", "A"])
    assert two["graph.scatter"] == {"y": "B", "x": "A", "by": None}    # the order of selection decides Y and X
    assert suggested(["A", "B"])["graph.scatter"]["y"] == "A"
    assert "graph.matrix" in two and "graph.contour" not in two
    grouped = suggested(["A", "G"])
    assert grouped["graph.boxplot"] == {"columns": ["A"], "by": "G"} and "graph.pie" not in grouped
    assert "graph.scatter" in suggested(["A", "B", "G"]) and "graph.contour" in suggested(["A", "B", "A"])   # three numeric columns
    assert set(suggested(["G"])) and all(a.startswith("graph.") for a in suggested(["G"]))
    assert suggested([]) == {}
    a = registry.get("basic.describe")
    assert assistant.preset(a, ws, ["A", "B", "G"]) == {"columns": ["A", "B"], "by": "G"}
    assert assistant.preset(registry.get("graph.contour"), ws, ["A", "B"]) is None


def test_conditional_formatting_rules(ws):
    a = ws.df["A"]
    assert formatting.rule_mask(a, formatting.make_rule("Greater than", "3")).tolist() == [
        False, False, False, True, True, False, True, False]
    assert formatting.rule_mask(a, formatting.make_rule("Less than", "2,5")).sum() == 2        # decimal comma
    assert formatting.rule_mask(a, formatting.make_rule("Equal to", "3")).tolist().index(True) == 2
    assert formatting.rule_mask(a, formatting.make_rule("Missing")).tolist() == [False] * 7 + [True]
    assert formatting.rule_mask(a, formatting.make_rule("Outlier (1.5 x IQR beyond the quartiles)")).tolist() == [
        False, False, False, False, True, False, False, False]
    g = ws.df["G"]
    assert formatting.rule_mask(g, formatting.make_rule("Equal to", "x")).sum() == 4
    assert formatting.rule_mask(g, formatting.make_rule("Not equal to", "x")).sum() == 3      # not the missing one
    assert not formatting.rule_mask(g, formatting.make_rule("Greater than", "3")).any()       # text against a number
    assert formatting.rule_mask(g, formatting.make_rule("Greater than", "x")).sum() == 3
    dates = pd.Series(pd.to_datetime(["2024-01-01", "2024-06-01"]))
    assert formatting.rule_mask(dates, formatting.make_rule("Greater than", "2024-03-01")).tolist() == [False, True]
    for bad in [("Greater than", ""), ("Bigger", "1"), ("Missing", "", "Pink")]:
        with pytest.raises(ValueError):
            formatting.make_rule(*bad)

    ws.formats["A"] = [formatting.make_rule("Greater than", "2", "Yellow"), formatting.make_rule("Greater than", "10")]
    colors = formatting.column_colors(ws, "A")
    assert colors[0] is None and colors[2] == formatting.COLORS["Yellow"] and colors[4] == formatting.COLORS["Red"]
    ws.rename_column(0, "Amount")
    assert list(ws.formats) == ["Amount"]


def test_formats_are_saved_and_shown(window, ws, tmp_path):
    from PySide6.QtCore import Qt

    view = window.current_view()
    view.set_formats("A", [formatting.make_rule("Greater than", "10", "Green")])
    model = view.model_
    assert model.data(model.index(4, 0), Qt.BackgroundRole).name() == formatting.COLORS["Green"]
    assert model.data(model.index(0, 0), Qt.BackgroundRole) is None and model.data(model.index(4, 1), Qt.BackgroundRole) is None
    model.set_cells([(0, 0, "99")])                                    # the highlight follows the data
    assert model.data(model.index(0, 0), Qt.BackgroundRole) is not None
    assert model.data(model.index(500, 0), Qt.BackgroundRole) is None  # the empty rows below the data
    window.project.save(tmp_path / "f.ostat")
    loaded = Project.load(tmp_path / "f.ostat").worksheets[-1]
    assert loaded.formats == ws.formats
    view.set_formats("A", [])
    assert "A" not in ws.formats and model.data(model.index(0, 0), Qt.BackgroundRole) is None


def test_assistant_and_graph_builder_dialogs(window, ws):
    from PySide6.QtWidgets import QPushButton

    from openstat.ui.assistant_dialog import AssistantDialog, GraphBuilderDialog
    from openstat.ui.dialog_builder import AnalysisDialog

    dialog = AssistantDialog(window)
    assert not dialog.back.isEnabled()
    node = assistant.TREE
    while not node.analysis:                                           # always take the first answer
        node = node.options[0][1]
        dialog.choose(node)
    assert dialog.back.isEnabled() and dialog.question.text() == registry.get(node.analysis).title
    dialog.go_back()
    assert dialog.options.count() == len(dialog.path[-1].options)
    dialog.choose(node)
    dialog.options.itemAt(1).widget().click()
    assert isinstance(dialog.options.itemAt(1).widget(), QPushButton) and dialog.chosen == "basic.graphical_summary"

    builder = GraphBuilderDialog(ws, window)
    assert builder.graphs.count() == 0
    builder.columns.item(1).setSelected(True)                          # B first: it becomes Y
    builder.columns.item(0).setSelected(True)
    assert builder.selected_columns() == ["B", "A"]
    titles = [builder.graphs.item(i).text() for i in range(builder.graphs.count())]
    builder.graphs.setCurrentRow(titles.index("Scatterplot"))
    builder.accept()
    a, values = builder.chosen
    assert a.id == "graph.scatter" and values["y"] == "B"
    filled = AnalysisDialog(a, ws, initial=values).values()
    assert filled["y"] == "B" and filled["x"] == "A" and filled["by"] is None
    preset = AnalysisDialog(registry.get("basic.describe"), ws, initial={"columns": ["B"], "by": "G", "more": True,
                                                                           "unknown": 1}).values()
    assert preset == {"columns": ["B"], "by": "G", "more": True}


def test_notes_selection_and_removal_of_output(window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "information", lambda *args: None)   # a modal box would wait forever
    console = window.console
    console.execute("run('basic.describe', columns=['A'])")
    window.add_note("Checked with the supplier.\nSecond line.")
    console.execute("run('basic.describe', columns=['B'])")
    assert [window.navigator.item(i).text() for i in range(3)] == ["Descriptive Statistics", "Note",
                                                                   "Descriptive Statistics"]
    assert "Second line." in window.output.toPlainText()
    window.navigator.clearSelection()
    window.export_output("html", str(tmp_path / "none.html"), selected=True)   # nothing selected: nothing written
    assert not (tmp_path / "none.html").exists()
    window.navigator.item(1).setSelected(True)
    assert window.selected_outputs() == [1]
    for kind in ("html", "docx", "pdf"):
        window.export_output(kind, str(tmp_path / f"sel.{kind}"), selected=True)
    html = (tmp_path / "sel.html").read_text(encoding="utf-8")
    assert "supplier" in html and "<table" not in html
    assert (tmp_path / "sel.pdf").read_bytes()[:5] == b"%PDF-" and (tmp_path / "sel.docx").stat().st_size > 0
    window.remove_selected_output()
    assert window.navigator.count() == 2 and "supplier" not in window.output.toPlainText()
    window.add_note("   ")
    assert window.navigator.count() == 2


def test_database_import(tmp_path):
    path = tmp_path / "plant.sqlite"
    link = sqlite3.connect(path)
    link.execute("CREATE TABLE parts (name TEXT, weight REAL, made TEXT)")
    link.executemany("INSERT INTO parts VALUES (?, ?, ?)", [("a", 1.5, "2024-01-02"), ("b", 2.5, "2024-01-03"),
                                                            ("c", None, "2024-01-04")])
    link.commit()
    link.close()
    ws = io.import_sql(str(path), "SELECT name, weight FROM parts WHERE weight > 1 ORDER BY name")
    assert ws.df["name"].tolist() == ["a", "b"] and ws.df["weight"].tolist() == [1.5, 2.5] and ws.is_numeric("weight")
    with pytest.raises(Exception):
        io.import_sql(str(path), "DELETE FROM parts")                 # the file is opened read-only
    assert len(io.import_sql(str(path), "SELECT * FROM parts")) if False else io.import_sql(
        str(path), "SELECT * FROM parts").nrows == 3
    for connection, query in [("", "SELECT 1"), (str(path), ""), (str(tmp_path / "missing.db"), "SELECT 1")]:
        with pytest.raises(ValueError):
            io.import_sql(connection, query)


def test_session_is_recorded_and_replayed(window, ws, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "information", lambda *args: None)
    a = registry.get("basic.t1")
    assert registry.format_call(a, {"columns": ["A"], "mu0": 2.0, "confidence": 95.0, "alternative": "Not equal"}) == \
        "run('basic.t1', columns=['A'], mu0=2.0)"                      # the defaults are left out
    merge = registry.get("data.merge")
    assert registry.format_call(merge, {"other": ws, "by": [], "rows": "All rows of both worksheets"}) == \
        "run('data.merge', other='Data')"
    window.save_session(str(tmp_path / "empty.py"))
    assert not (tmp_path / "empty.py").exists()                        # nothing to save yet

    window.console.execute("run('basic.t1', columns=['A'], mu0=2.0)")
    window.console.execute("run('graph.scatter', y='B', x='A')")
    assert window.session == ["run('basic.t1', columns=['A'], mu0=2.0)", "run('graph.scatter', y='B', x='A')"]
    script = tmp_path / "session.py"
    window.save_session(str(script))
    assert script.read_text(encoding="utf-8").splitlines()[1:] == window.session

    count = len(window.project.outputs)
    window.run_script(str(script))
    assert len(window.project.outputs) == count + 2 and len(window.session) == 4
    assert window.navigator.item(count).text() == "1-Sample t"
    broken = tmp_path / "broken.py"
    broken.write_text("print('before')\nundefined_name\n", encoding="utf-8")
    window.run_script(str(broken))                                     # the error goes to the command line
    log = window.console.log.toPlainText()
    assert "before" in log and "NameError" in log
    assert window.console.run_script("df['C'] = df['A'] * 2") == "" and ws.df["C"].iloc[1] == 4.0
