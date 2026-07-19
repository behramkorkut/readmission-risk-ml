"""Tests de l'utilité clinique (courbe de décision).

On vérifie le comportement du calcul délégué à `dcurves` contre les propriétés
théoriques du bénéfice net — pas contre des valeurs magiques.
"""

import numpy as np
import pytest

from readmission_risk.evaluation.clinical_utility import compute_decision_curve, summarize


def _curve(y, p):
    return compute_decision_curve(np.asarray(y), np.asarray(p))


def test_ne_traiter_personne_vaut_toujours_zero():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 2, 200), rng.random(200)
    none = _curve(y, p).query("model == 'none'")["net_benefit"]
    assert np.allclose(none, 0.0)


def test_traiter_tout_le_monde_suit_la_formule_theorique():
    # NB_all(t) = prévalence − (1 − prévalence) · t/(1−t)
    y = np.array([1] * 20 + [0] * 80)  # prévalence 0.20
    c = _curve(y, np.linspace(0, 1, 100))
    for t in (0.10, 0.20, 0.30):
        nb = float(c[(c["model"] == "all") & np.isclose(c["threshold"], t)]["net_benefit"].iloc[0])
        assert nb == pytest.approx(0.2 - 0.8 * t / (1 - t), abs=1e-9)


def test_un_modele_parfait_domine_les_baselines():
    y = np.array([1] * 30 + [0] * 70)  # prévalence 0.30
    p = np.where(y == 1, 0.95, 0.05)  # séparation parfaite
    piv = _curve(y, p).pivot(index="threshold", columns="model", values="net_benefit")
    zone = piv.loc[(piv.index >= 0.10) & (piv.index <= 0.50)]
    assert (zone["model"] >= zone["all"] - 1e-9).all()
    assert (zone["model"] >= zone["none"] - 1e-9).all()
    # Modèle parfait, seuil sous 0.95 : TP/n = prévalence, FP = 0 -> NB = 0.30
    assert float(piv.loc[np.isclose(piv.index, 0.20), "model"].iloc[0]) == pytest.approx(0.30, abs=1e-9)


def test_summarize_structure():
    y = np.array([0, 1] * 50)
    tab = summarize(_curve(y, np.linspace(0, 1, 100)))
    assert list(tab.columns) == ["seuil", "model", "all", "none"]
    assert len(tab) == 3
