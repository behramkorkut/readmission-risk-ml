"""Tests des utilitaires de calibration / conformal."""

import numpy as np
import pandas as pd

from readmission_risk.modeling.calibrate import coverage_and_size, three_way_split


def test_coverage_and_size():
    # 3 exemples, 2 classes, 1 niveau de confiance (forme MAPIE (n, 2, 1)).
    # ex0: ensemble {0,1} ; ex1: {1} ; ex2: {0}
    y_set = np.array([
        [[1], [1]],
        [[0], [1]],
        [[1], [0]],
    ])
    y_true = np.array([0, 1, 1])  # ex2 mal couvert (vraie classe 1 absente)
    cov, size = coverage_and_size(y_set, y_true)
    assert cov == 2 / 3                 # ex0 et ex1 couverts, pas ex2
    assert size == (2 + 1 + 1) / 3      # tailles 2,1,1


def test_three_way_split_disjoint_par_patient():
    rng = np.random.default_rng(0)
    rows = []
    for pid in range(400):
        for _ in range(rng.integers(1, 4)):
            rows.append({"patient_nbr": pid, "y": int(rng.random() < 0.2)})
    df = pd.DataFrame(rows)
    fit, calib, conform = three_way_split(df, "y", "patient_nbr", seed=42)

    pf, pc, pk = set(fit["patient_nbr"]), set(calib["patient_nbr"]), set(conform["patient_nbr"])
    assert pf & pc == set() and pf & pk == set() and pc & pk == set()  # 3 jeux disjoints
    assert len(fit) + len(calib) + len(conform) == len(df)
