import numpy as np
import pandas as pd
import pytest

from openstat import i18n
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


def test_date_cells():
    assert parse_cell("2024-02-29") == pd.Timestamp("2024-02-29")
    assert parse_cell("2024-02-29 08:30") == pd.Timestamp("2024-02-29 08:30")
    assert parse_cell("2023-02-29") == "2023-02-29"       # not a real date: stays text
    assert parse_cell("1-2") == "1-2"
    i18n.set_language("pt_BR")
    try:
        assert parse_cell("05/03/2024") == pd.Timestamp("2024-03-05")
    finally:
        i18n.set_language("en")
    assert parse_cell("05/03/2024") == pd.Timestamp("2024-05-03")

    ws = Worksheet()
    ws.set_cell(0, 0, "2024-01-05")
    ws.set_cell(2, 0, "2024-01-07 12:00:00")
    assert ws.is_datetime("C1")
    assert [ws.get_text(r, 0) for r in range(3)] == ["2024-01-05", "", "2024-01-07 12:00:00"]
    ws.set_cell(1, 0, "abc")
    assert not ws.is_datetime("C1") and ws.get_text(0, 0) == "2024-01-05"
    ws.set_cell(1, 0, "")
    assert ws.is_datetime("C1")


def test_formulas(tmp_path):
    ws = Worksheet(df=pd.DataFrame({"A": [1.0, 2.0], "B": [10.0, 20.0]}))
    ws.set_formula("Total", "A + B")
    ws.set_formula("Twice", "Total * 2")
    assert ws.df["Twice"].tolist() == [22.0, 44.0]
    ws.set_cell(0, 0, "5")
    assert ws.df["Total"].tolist() == [15.0, 22.0] and ws.df["Twice"].tolist() == [30.0, 44.0]
    ws.set_cell(2, 1, "30")                                  # a new row is filled too
    assert np.isnan(ws.df["Total"].iloc[2]) and ws.nrows == 3
    ws.set_formula("Mean", "mean(B)")                        # a constant fills the column
    assert ws.df["Mean"].tolist() == [20.0, 20.0, 20.0]
    with pytest.raises(ValueError):
        ws.set_formula("Total", "total + 1")
    with pytest.raises(ValueError):
        ws.set_formula("Bad", "Nope + 1")
    assert "Bad" not in ws.column_names()

    ws.rename_column(3, "Double")
    assert ws.formulas["Double"] == "Total * 2"
    ws.rename_column(0, "First")                             # the formula loses its column: values stay
    assert "Total" in ws.formula_errors and ws.df["Total"].tolist()[:2] == [15.0, 22.0]
    ws.rename_column(0, "A")
    assert not ws.formula_errors

    project = Project(worksheets=[ws])
    project.save(tmp_path / "f.ostat")
    loaded = Project.load(tmp_path / "f.ostat").worksheets[0]
    assert loaded.formulas == ws.formulas
    loaded.set_cell(1, 0, "0")
    assert loaded.df["Total"].tolist()[:2] == [15.0, 20.0]
    loaded.set_formula("Mean", "")
    assert "Mean" not in loaded.formulas and loaded.df["Mean"].tolist() == [20.0, 20.0, 20.0]


def test_project_keeps_dates(tmp_path):
    ws = Worksheet("D", pd.DataFrame({"When": pd.to_datetime(["2024-03-15", None])}))
    Project(worksheets=[ws]).save(tmp_path / "d.ostat")
    loaded = Project.load(tmp_path / "d.ostat").worksheets[0]
    assert loaded.is_datetime("When") and loaded.get_text(0, 0) == "2024-03-15" and loaded.get_text(1, 0) == ""


def test_import_csv_dates(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("day,label,n\n2024-01-05,2024-01-05,1\n2024-01-06,x,2\n", encoding="utf-8")
    ws = io.import_file(path)[0]
    assert ws.is_datetime("day") and not ws.is_datetime("label")
    assert ws.df["day"].tolist() == [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-06")]


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
