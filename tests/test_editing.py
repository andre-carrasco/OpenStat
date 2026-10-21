"""Worksheet editing: snapshots for undo, column properties and their storage in the project."""
import json
import zipfile

import pandas as pd

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
