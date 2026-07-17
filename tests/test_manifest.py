"""Tests du manifeste de traçabilité des données (audit n°3).

Roundtrip record -> verify, détection de modification (tampering) et d'absence,
déterminisme du hash, et auto-enregistrement lors de l'ingestion (réseau mocké).
Tout est exécuté dans tmp_path : ni le vrai data/, ni le vrai manifeste ne sont touchés.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from readmission_risk.common.config import settings
from readmission_risk.data import ingest as ingest_mod
from readmission_risk.data.manifest import (
    load_manifest,
    main,
    record_artifact,
    sha256_file,
    verify,
)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Redirige data_dir (artefacts + manifeste) vers tmp_path."""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return tmp_path


def _write_parquet(path, values):
    pd.DataFrame({"a": values}).to_parquet(path, index=False)


def test_hash_deterministic(data_dir):
    f = data_dir / "x.bin"
    f.write_bytes(b"contenu stable")
    assert sha256_file(f) == sha256_file(f)  # déterministe
    g = data_dir / "y.bin"
    g.write_bytes(b"contenu stable")
    assert sha256_file(f) == sha256_file(g)  # même contenu -> même hash
    g.write_bytes(b"autre contenu")
    assert sha256_file(f) != sha256_file(g)


def test_record_and_verify_ok(data_dir):
    f = data_dir / "table.parquet"
    _write_parquet(f, [1, 2, 3])
    entry = record_artifact(f, produced_by="readmission-clean")
    # Contrat de l'entrée : hash + dimensions + producteur + traçabilité code.
    assert len(entry["sha256"]) == 64
    assert entry["rows"] == 3 and entry["cols"] == 1
    assert entry["produced_by"] == "readmission-clean"
    assert entry["produced_at_utc"].endswith("+00:00")
    assert "code_git_sha" in entry
    # Le manifeste est un JSON versionnable, lisible par un humain.
    on_disk = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["artifacts"]["table.parquet"]["sha256"] == entry["sha256"]
    assert verify() == [("table.parquet", "OK", "3 lignes, produit par readmission-clean")]


def test_verify_detects_tampering(data_dir):
    f = data_dir / "table.parquet"
    _write_parquet(f, [1, 2, 3])
    record_artifact(f, produced_by="readmission-clean")
    _write_parquet(f, [9, 9, 9, 9])  # même nom, contenu différent -> dérive
    [(name, status, detail)] = verify()
    assert name == "table.parquet"
    assert status == "MODIFIÉ"
    assert "≠" in detail


def test_verify_detects_missing(data_dir):
    f = data_dir / "table.parquet"
    _write_parquet(f, [1])
    record_artifact(f, produced_by="readmission-clean")
    f.unlink()
    assert verify()[0][1] == "ABSENT"


def test_verify_ignores_untracked_files(data_dir):
    # Un fichier non enregistré (ex. predictions_log.db) ne doit pas fausser la vérif.
    _write_parquet(data_dir / "table.parquet", [1])
    record_artifact(data_dir / "table.parquet", produced_by="readmission-clean")
    (data_dir / "predictions_log.db").write_bytes(b"journal de monitoring")
    assert [status for _, status, _ in verify()] == ["OK"]


def test_main_exit_code_on_drift(data_dir, capsys):
    _write_parquet(data_dir / "table.parquet", [1])
    record_artifact(data_dir / "table.parquet", produced_by="readmission-clean")
    main()  # tout OK : pas d'exception
    assert "correspondent au manifeste" in capsys.readouterr().out
    _write_parquet(data_dir / "table.parquet", [2, 3])  # dérive
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1  # utilisable en CI


def test_ingest_auto_records_manifest(data_dir, monkeypatch):
    monkeypatch.setattr(
        ingest_mod, "download_raw", lambda: pd.DataFrame({"encounter_id": [1, 2], "readmitted": ["NO", "<30"]})
    )
    ingest_mod.ingest(force=True)  # réseau mocké
    manifest = load_manifest()
    assert settings.raw_filename in manifest["artifacts"]
    assert manifest["artifacts"][settings.raw_filename]["produced_by"] == "readmission-ingest"
    assert manifest["artifacts"][settings.raw_filename]["rows"] == 2


def test_ingest_skip_backfills_manifest(data_dir, monkeypatch):
    # Artefact produit AVANT l'introduction du manifeste : skip, mais enregistré.
    _write_parquet(data_dir / settings.raw_filename, [1, 2, 3])
    monkeypatch.setattr(ingest_mod, "download_raw", lambda: pytest.fail("réseau appelé !"))
    ingest_mod.ingest(force=False)  # skip (fichier présent), aucun téléchargement
    assert load_manifest()["artifacts"][settings.raw_filename]["rows"] == 3
    # Déjà enregistré : un second skip ne réécrit pas le manifeste (pas de bruit git).
    before = (data_dir / "manifest.json").read_text(encoding="utf-8")
    ingest_mod.ingest(force=False)
    assert (data_dir / "manifest.json").read_text(encoding="utf-8") == before
