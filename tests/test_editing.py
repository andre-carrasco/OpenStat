"""Worksheet editing: snapshots for undo, column properties and their storage in the project."""
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from openstat.core.project import Project
from openstat.core.worksheet import Worksheet, changed_parts


def ws_ab():
    return Worksheet("W", pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": ["x", "y", "x"]}))


def test_snapshot_restore_roundtrip():
    ws = ws_ab(); before = ws.snapshot()
    ws.set_cell(5, 3, "7"); ws.formats["A"] = [{"rule": "x"}]; ws.descriptions["A"] = "d"
    ws.restore(before)
    assert ws.column_names() == ["A", "B"] and ws.nrows == 3 and ws.formats == {} and ws.descriptions == {}


def test_changed_parts_keeps_only_changed_columns():
    ws = ws_ab(); before = ws.snapshot(); ws.set_cell(0, 0, "9"); after = ws.snapshot()
    b, a = changed_parts(before, after)
    assert set(b["data"]) == {"A"} and set(a["data"]) == {"A"}
    ws.restore(b); assert ws.df["A"].tolist() == [1.0, 2.0, 3.0] and ws.df["B"].tolist() == ["x", "y", "x"]
    ws.restore(a); assert ws.df["A"].tolist() == [9.0, 2.0, 3.0]


def test_rename_carries_properties_and_restores():
    ws = ws_ab(); ws.set_formula("F", "A * 2"); ws.descriptions["A"] = "d"; ws.value_orders["B"] = ["y", "x"]
    before = ws.snapshot(); ws.rename_column(1, "Bee")
    assert ws.value_orders == {"Bee": ["y", "x"]}
    ws.restore(changed_parts(before, ws.snapshot())[0])
    assert ws.column_names() == ["A", "B", "F"] and ws.value_orders == {"B": ["y", "x"]} and ws.formulas == {"F": "A * 2"}


def test_project_saves_descriptions_and_value_orders(tmp_path):
    p = Project([ws_ab()]); p.worksheets[0].descriptions["A"] = "d"; p.worksheets[0].value_orders["B"] = "occurrence"
    p.save(tmp_path / "p.ostat"); ws = Project.load(tmp_path / "p.ostat").worksheets[0]
    assert ws.descriptions == {"A": "d"} and ws.value_orders == {"B": "occurrence"}


def test_old_project_without_new_keys_opens(tmp_path):
    path, old = tmp_path / "p.ostat", tmp_path / "old.ostat"
    Project([ws_ab()]).save(path)
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(old, "w") as dst:
        for item in src.namelist():
            data = src.read(item)
            if item == "manifest.json":
                manifest = json.loads(data)
                manifest.pop("descriptions"); manifest.pop("value_orders")
                data = json.dumps(manifest).encode("utf-8")
            dst.writestr(item, data)
    ws = Project.load(old).worksheets[0]
    assert ws.descriptions == {} and ws.value_orders == {}


def test_delete_cells_shifts_up_per_column():
    ws = ws_ab(); ws.delete_cells([(0, 0)])
    assert ws.df["A"].tolist()[:2] == [2.0, 3.0] and np.isnan(ws.df["A"][2]) and ws.df["B"].tolist() == ["x", "y", "x"]

def test_clear_cells_leaves_blanks():
    ws = ws_ab(); ws.clear_cells([(1, 0), (1, 1)])
    assert np.isnan(ws.df["A"][1]) and ws.df["B"][1] is None and ws.nrows == 3

def test_insert_cells_pushes_down():
    ws = ws_ab(); ws.insert_cells([(0, 0), (1, 0)])
    assert ws.nrows == 5 and ws.df["A"].tolist()[2:] == [1.0, 2.0, 3.0]

def test_insert_and_delete_rows_and_columns():
    ws = ws_ab(); ws.insert_rows(1, 2); assert ws.nrows == 5 and ws.df["A"].tolist()[3:] == [2.0, 3.0]
    ws.delete_rows([1, 2]); assert ws.df["A"].tolist() == [1.0, 2.0, 3.0]
    ws.insert_columns(1, 1); assert ws.column_names()[0] == "A" and ws.column_names()[2] == "B"
    ws.descriptions["B"] = "d"; ws.delete_columns([2]); assert "B" not in ws.column_names() and ws.descriptions == {}

def test_move_columns():
    ws = Worksheet("W", pd.DataFrame({c: [1.0] for c in "ABCD"}))
    ws.move_columns(["D", "B"], "start"); assert ws.column_names() == ["B", "D", "A", "C"]
    ws.move_columns(["B"], "after", "C"); assert ws.column_names() == ["D", "A", "C", "B"]
    with pytest.raises(ValueError): ws.move_columns(["A"], "before", "A")

def test_text_column_becomes_numeric_after_deleting_its_text():
    ws = Worksheet("W", pd.DataFrame({"A": ["a", "2", "3"]})); ws.delete_cells([(0, 0)])
    assert ws.is_numeric("A")

def test_insert_rows_beyond_the_end_of_the_data_still_inserts_count():
    ws = ws_ab(); ws.insert_rows(2, 2)
    assert ws.nrows == 5 and ws.df["A"].tolist()[:2] == [1.0, 2.0] and ws.df["A"].isna().tolist()[2:4] == [True, True] and ws.df["A"][4] == 3.0
    ws = ws_ab(); ws.insert_rows(0, 5); assert ws.nrows == 8

def test_insert_rows_without_columns_is_a_no_op():
    ws = ws_ab(); ws.delete_columns([0, 1]); ws.insert_rows(0, 1)
    assert ws.ncols == 0


def test_levels_default_orders():
    ws = Worksheet("W", pd.DataFrame({"N": [10.0, 2.0, 1.0], "T": ["b", "a", "c"],
                                      "D": pd.to_datetime(["2024-02-01", "2023-12-31", "2024-01-15"])}))
    assert ws.levels("N") == ["1", "2", "10"] and ws.levels("T") == ["a", "b", "c"]
    assert ws.levels("D") == ["2023-12-31", "2024-01-15", "2024-02-01"]


def test_levels_occurrence_and_user_order():
    ws = Worksheet("W", pd.DataFrame({"T": ["b", "a", "c", "a"]}))
    ws.value_orders["T"] = "occurrence"
    assert ws.levels("T") == ["b", "a", "c"]
    ws.value_orders["T"] = ["c", "z", "a"]
    assert ws.levels("T") == ["c", "a", "b"]   # z missing, b new at end
    assert ws.levels("T", pd.Series(["a", "c"])) == ["c", "a"]


def test_levels_are_distinct():
    ws = Worksheet("W", pd.DataFrame({"N": [1.00000000001, 1.00000000002, 2.0], "T": ["a", "b", "a"]}))
    assert ws.levels("N") == ["1", "2"]
    ws.value_orders["N"] = "occurrence"
    assert ws.levels("N") == ["1", "2"]
    ws.value_orders["T"] = ["b", "b", "a"]
    assert ws.levels("T") == ["b", "a"]


def test_levels_label_missing_from_column_goes_last():
    ws = Worksheet("W", pd.DataFrame({"T": ["b", "a"]}))
    assert ws.levels("T", pd.Series(["z", "a", "y"])) == ["a", "y", "z"]
    ws.value_orders["T"] = "occurrence"
    assert ws.levels("T", pd.Series(["z", "a", "b", "y"])) == ["b", "a", "y", "z"]
