import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from openstat.core.worksheet import Worksheet


@pytest.fixture
def sample_ws() -> Worksheet:
    rng = np.random.default_rng(42)
    n = 40
    x = rng.normal(50, 5, n)
    mix = rng.dirichlet([2, 2, 2], n)
    return Worksheet("Sample", pd.DataFrame({
        "Y": 3 + 0.8 * x + rng.normal(0, 2, n),
        "X": x,
        "Z": rng.normal(10, 1, n),
        "Count": rng.poisson(3, n).astype(float),
        "Group": np.array(["A", "B", "C", "D"], dtype=object)[np.arange(n) % 4],
        "Pass": np.array(["no", "yes"], dtype=object)[(rng.random(n) < 0.7).astype(int)],
        "Block": np.array([f"b{i}" for i in range(10)], dtype=object)[np.arange(n) // 4],
        "Seq": np.array(["AB", "BA"], dtype=object)[np.arange(n) % 2],
        # two batches nested within each group, five observations per batch
        "Batch": np.array([f"{'ABCD'[i % 4]}{(i // 4) % 2}" for i in range(n)], dtype=object),
        # balanced crossed gage layout: 5 parts x 2 operators x 4 replicates
        "Part": np.array([f"p{i % 5}" for i in range(n)], dtype=object),
        "Operator": np.array([f"o{(i // 5) % 2}" for i in range(n)], dtype=object),
        # mixture components summing to 1
        "M1": mix[:, 0], "M2": mix[:, 1], "M3": mix[:, 2],
    }))


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])
