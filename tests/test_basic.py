import numpy as np
import pytest
from scipy import stats as sps

from openstat.stats.basic import describe, describe_values


def test_describe_values_known():
    # 1..9: quartiles by the (n+1)p rule are 2.5, 5, 7.5
    d = describe_values([1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan])
    assert d["N"] == 9 and d["N*"] == 1
    assert d["Mean"] == pytest.approx(5.0)
    assert d["StDev"] == pytest.approx(np.std(np.arange(1, 10), ddof=1))
    assert d["SE Mean"] == pytest.approx(sps.sem(np.arange(1, 10)))
    assert (d["Q1"], d["Median"], d["Q3"]) == pytest.approx((2.5, 5.0, 7.5))
    assert (d["Minimum"], d["Maximum"]) == (1.0, 9.0)


def test_describe_empty():
    d = describe_values([np.nan])
    assert d["N"] == 0 and np.isnan(d["Mean"])


def test_describe_by_group(sample_ws):
    res = describe(sample_ws, ["Y"], by="Group")
    table = res.blocks[0][1][0]
    assert len(table) == 4
    assert table["N"].sum() == 40
