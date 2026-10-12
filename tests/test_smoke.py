"""Runs every registered analysis with default arguments, through the generated dialog."""
import pytest

from openstat import i18n
from openstat.core import registry
from openstat.core.result import Result

ANALYSES = registry.all_analyses()


# Values for parameters that have no usable default
OVERRIDES = {
    "data.subset": {"condition": "Y > mean(Y)"},
    "data.recode": {"mapping": "A=1; B=2"},
    "calc.calculator": {"expression": "Y + 2 * X"},
    "basic.p1": {"column": "Pass"},
    "basic.p2": {"events1": 12, "trials1": 40, "events2": 20, "trials2": 45},
    "basic.poisson1": {"column": "Count"},
    "basic.gof_poisson": {"column": "Count"},
    "tables.crosstab": {"rows": "Group", "cols": "Pass"},
    "nonpar.friedman": {"treatment": "Group", "block": "Block"},
    "equiv.crossover": {"sequence": "Seq"},
    "reg.logistic": {"response": "Pass"},
    "reg.poisson": {"response": "Count"},
    "reg.ordinal": {"categorical": []},
    "reg.nominal": {"categorical": []},
    "anova.nested": {"factor_a": "Group", "factor_b": "Batch"},
    "anova.interaction": {"factor1": "Group", "factor2": "Pass"},
    "quality.capability_normal": {"lsl": "30", "usl": "60", "target": "45"},
    "quality.capability_nonnormal": {"lsl": "30", "usl": "60"},
    "quality.sixpack": {"lsl": "30", "usl": "60"},
    "gage.crossed": {"part": "Part", "operator": "Operator", "tolerance": "20"},
    "gage.nested": {"part": "Part", "operator": "Operator"},
    "gage.attribute": {"rating": "Pass", "sample": "Part", "appraiser": "Operator"},
    # defective counts must not exceed the subgroup size
    "cc.p": {"column": "Count"},
    "cc.np": {"column": "Count"},
    "cc.laney_p": {"column": "Count"},
    "quality.capability_binomial": {"column": "Count"},
    "doe.analyze_factorial": {"factors": ["Pass", "Seq", "Operator"]},
    "doe.analyze_mixture": {"components": ["M1", "M2", "M3"]},
    "rel.arbitrary": {"start": "Start", "end": "End"},
    "rel.probit": {"events": "Events", "trials": "Trials", "stress": "Dose"},
    "mv.cluster_var": {"columns": ["Y", "X", "Z"]},
    "mv.multiple_correspondence": {"columns": ["Group", "Pass"]},
}


def default_values(a, ws):
    """Dialog values with sensible column picks for a generic worksheet."""
    numeric = ws.numeric_columns()
    text = [c for c in ws.column_names() if c not in numeric]
    values, used = {}, 0
    for p in a.params:
        if p.kind == "columns":
            values[p.name] = [numeric[(used + i) % len(numeric)] for i in range(2)] if p.numeric else text[:1]
            used += 2 if p.numeric else 0
        elif p.kind == "column":
            if p.optional:
                values[p.name] = None
            elif p.numeric:
                values[p.name] = numeric[used % len(numeric)]
                used += 1
            else:
                values[p.name] = text[0]
    values.update(OVERRIDES.get(a.id, {}))
    return values


@pytest.mark.parametrize("a", ANALYSES, ids=[a.id for a in ANALYSES])
def test_analysis_runs(a, sample_ws, qapp):
    from openstat.ui.dialog_builder import AnalysisDialog

    dialog = AnalysisDialog(a, sample_ws)
    values = dialog.values()
    values.update(default_values(a, sample_ws))
    result = a.func(sample_ws, **values)
    assert isinstance(result, Result)
    assert result.blocks
    assert "<h2" in result.to_html(anchor="r0")


@pytest.mark.parametrize("lang", ["en", "pt_BR"])
def test_main_window(qapp, sample_ws, lang, tmp_path):
    from openstat.ui.main_window import MainWindow

    i18n.set_language(lang)
    try:
        window = MainWindow()
        window.add_worksheet(sample_ws)
        for a in ANALYSES:
            window.add_result(a.func(sample_ws, **{**AnalysisDialog_defaults(a, sample_ws)}), sample_ws)
        assert window.navigator.count() == len(ANALYSES)
        assert window.output.toPlainText().strip()
        window.project.save(tmp_path / "p.ostat")
        window.grab().save(str(tmp_path / "window.png"))
    finally:
        i18n.set_language("en")


def AnalysisDialog_defaults(a, ws):
    from openstat.ui.dialog_builder import AnalysisDialog

    values = AnalysisDialog(a, ws).values()
    values.update(default_values(a, ws))
    return values
