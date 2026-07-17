"""Manifeste de traçabilité des artefacts de données (audit n°3 — alternative légère à DVC).

Chaque étape productrice (ingest -> brut, clean -> nettoyé) y auto-enregistre son
artefact : hash SHA-256 (MD5 suffirait pour de l'intégrité, mais SHA-256 n'a pas de
collisions connues et coûte le même prix), dimensions (lues dans les métadonnées
Parquet, sans charger les données), producteur (commande CLI), date, et SHA git du
code — la traçabilité code <-> donnée que réclamait l'audit.

Le manifeste (`data/manifest.json`) est VERSIONNÉ dans git (exception .gitignore) :
c'est lui qui apporte la traçabilité ; les Parquets, eux, restent non versionnés.
La CLI `readmission-data-verify` recalcule les hash et signale toute dérive
(artefact modifié ou absent) — utilisable en CI (code de sortie 1).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import structlog

from readmission_risk.common.config import PROJECT_ROOT, settings

log = structlog.get_logger()

MANIFEST_FILENAME = "manifest.json"


def _manifest_path() -> Path:
    return settings.data_dir / MANIFEST_FILENAME


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 du fichier, lu par blocs (les Parquets peuvent être volumineux)."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_short_sha() -> str | None:
    """SHA court du commit courant (traçabilité code), None hors dépôt git."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def load_manifest() -> dict:
    """Charge le manifeste (squelette vide s'il n'existe pas encore)."""
    path = _manifest_path()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"version": 1, "artifacts": {}}


def record_artifact(path: Path, produced_by: str) -> dict:
    """Enregistre (ou met à jour) un artefact dans le manifeste. Renvoie l'entrée."""
    meta = pq.read_metadata(path)  # dimensions sans charger le Parquet
    entry = {
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "rows": meta.num_rows,
        "cols": meta.num_columns,
        "produced_by": produced_by,
        "produced_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_git_sha": git_short_sha(),
    }
    manifest = load_manifest()
    manifest["artifacts"][path.name] = entry
    _manifest_path().write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info("manifest.recorded", file=path.name, sha256=entry["sha256"][:12], rows=entry["rows"])
    return entry


def verify() -> list[tuple[str, str, str]]:
    """Recalcule les hash et compare au manifeste.

    Renvoie (nom, statut, détail) par artefact ; statut ∈ OK / MODIFIÉ / ABSENT.
    """
    results: list[tuple[str, str, str]] = []
    for name, entry in load_manifest()["artifacts"].items():
        path = settings.data_dir / name
        if not path.exists():
            results.append((name, "ABSENT", "fichier manquant"))
        elif (current := sha256_file(path)) != entry["sha256"]:
            results.append((name, "MODIFIÉ", f"hash actuel {current[:12]}… ≠ manifeste {entry['sha256'][:12]}…"))
        else:
            results.append((name, "OK", f"{entry['rows']} lignes, produit par {entry['produced_by']}"))
    return results


def main() -> None:
    """CLI `readmission-data-verify` : vérifie l'intégrité des artefacts de données."""
    results = verify()
    if not results:
        print("Manifeste vide — lance d'abord readmission-ingest puis readmission-clean.")
        sys.exit(1)
    print("\n=== Vérification des artefacts de données (manifeste SHA-256) ===")
    drift = False
    for name, status, detail in results:
        drift = drift or status != "OK"
        print(f"{status:8} {name}  {detail}")
    if drift:
        print("\n⚠ Dérive détectée : régénère les données (ingest -> clean) ou vérifie leur provenance.")
        sys.exit(1)
    print("\nTous les artefacts correspondent au manifeste.")


if __name__ == "__main__":
    main()
