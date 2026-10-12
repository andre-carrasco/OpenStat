import numpy as np
import pandas as pd

from openstat.core import io
from openstat.core.project import OutputItem, Project
from openstat.core.worksheet import Worksheet, parse_cell


def test_parse_cell():
    assert parse_cell("") is None
    assert parse_cell("*") is None
    assert parse_cell("1.5") == 1.5
    assert parse_cell("1,5") == 1.5
    assert parse_cell("abc") == "abc"


def test_set_cell_grows_and_types():
    ws = Worksheet()
    ws.set_cell(2, 1, "3.5")
    assert (ws.nrows, ws.ncols) == (3, 2)
    assert ws.column_names() == ["C1", "C2"]
    assert ws.is_numeric("C2")
    assert ws.get_text(2, 1) == "3.5"
    assert ws.get_text(0, 1) == ""

    ws.set_cell(0, 1, "abc")
    assert not ws.is_numeric("C2")
    assert ws.get_text(2, 1) == "3.5"

    ws.set_cell(0, 1, "")
    assert ws.is_numeric("C2")


def test_blank_outside_does_not_grow():
    ws = Worksheet()
    ws.set_cell(5, 5, "")
    assert (ws.nrows, ws.ncols) == (0, 0)


def test_rename_and_add_column():
    ws = Worksheet(df=pd.DataFrame({"C1": [1.0, 2.0]}))
    ws.rename_column(0, "Weight")
    assert ws.column_names() == ["Weight"]
    used = ws.add_column("Weight", [1, 2, 3])
    assert used == "Weight_2"
    assert ws.nrows == 3
    assert np.isnan(ws.df["Weight"].iloc[2])


def test_project_roundtrip(tmp_path):
    project = Project()
    ws = project.new_worksheet()
    ws.set_cell(0, 0, "1")
    ws.set_cell(1, 0, "2.5")
    ws.set_cell(0, 1, "txt")
    project.outputs.append(OutputItem("T", "<h2>T</h2>"))
    path = tmp_path / "p.ostat"
    project.save(path)

    loaded = Project.load(path)
    assert loaded.worksheets[0].name == ws.name
    assert loaded.worksheets[0].df["C1"].tolist() == [1.0, 2.5]
    assert loaded.worksheets[0].df["C2"].tolist()[0] == "txt"
    assert not loaded.worksheets[0].is_numeric("C2")
    assert loaded.outputs[0].html == "<h2>T</h2>"


def test_import_csv_semicolon_decimal_comma(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("a;b;c\n1,5;x;3\n2,5;y;4\n", encoding="utf-8")
    ws = io.import_file(path)[0]
    assert ws.df["a"].tolist() == [1.5, 2.5]
    assert ws.df["b"].tolist() == ["x", "y"]
    assert ws.is_numeric("c")


def test_import_xlsx(tmp_path):
    path = tmp_path / "d.xlsx"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_excel(path, index=False, sheet_name="S1")
    ws = io.import_file(path)[0]
    assert ws.name == "S1"
    assert ws.df["a"].tolist() == [1.0, 2.0]
