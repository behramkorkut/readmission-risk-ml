"""Ingestion reproductible du dataset Diabetes 130-US (UCI id=296).

On récupère la table COMPLÈTE via `ds.data.original` (et non `features`, qui
exclut les identifiants encounter_id / patient_nbr — indispensables à la
stratégie anti-fuite). On sauvegarde en Parquet (format colonnaire compact).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import structlog
from ucimlrepo import fetch_ucirepo

from readmission_risk.common.config import settings

log = structlog.get_logger()

UCI_ID = 296  # Diabetes 130-US hospitals for years 1999-2008


def download_raw() -> pd.DataFrame:
    """Télécharge la table brute complète depuis UCI."""
    ds = fetch_ucirepo(id=UCI_ID)
    # `original` = IDs + features + cible (50 colonnes). `features` en exclut 2.
    return ds.data.original


def ingest(force: bool = False) -> Path:
    """Télécharge et sauvegarde en Parquet. Idempotent (skip si déjà présent)."""
    out = settings.data_dir / settings.raw_filename
    if out.exists() and not force:
        log.info("ingest.skip", path=str(out))
        return out
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    df = download_raw()
    df.to_parquet(out, index=False)
    log.info("ingest.done", path=str(out), rows=len(df), cols=df.shape[1])
    return out


def main() -> None:
    ingest(force=False)


if __name__ == "__main__":
    main()
