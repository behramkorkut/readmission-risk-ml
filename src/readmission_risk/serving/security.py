"""Sécurité de l'API : authentification par clé + limitation de débit.

**Auth (clé API).** Dès que `settings.api_key` est défini (variable d'env `API_KEY`),
`/predict` exige l'en-tête `X-API-Key` : 401 si absente, 403 si invalide. La
comparaison se fait en temps constant (`hmac.compare_digest`) pour ne pas fuiter
la clé via les temps de réponse (timing attack). Tant qu'aucune clé n'est
configurée, l'accès reste libre — choix assumé pour la démo publique (Streamlit) ;
`/health` reste public dans tous les cas (sondes de supervision).

**Rate limiting.** Fenêtre glissante en mémoire, par IP client : au-delà de
`settings.rate_limit_per_minute` requêtes / 60 s -> 429 + en-tête `Retry-After`
(0 = désactivé). Limitation assumée : l'état est par processus — correct pour un
déploiement mono-worker (notre VPS) ; en multi-workers il faudrait un backend
partagé (Redis). Derrière un reverse proxy (nginx), l'IP vue est celle du proxy :
lancer uvicorn avec `--proxy-headers` pour propager l'IP réelle.
"""

from __future__ import annotations

import hmac
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from readmission_risk.common.config import settings

API_KEY_HEADER = "X-API-Key"
_api_key_scheme = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def require_api_key(api_key: str | None = Security(_api_key_scheme)) -> None:
    """Exige la clé API si elle est configurée (sinon accès libre — démo)."""
    if settings.api_key is None:
        return  # auth désactivée : aucune clé configurée (démo publique)
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Clé API manquante (en-tête {API_KEY_HEADER}).",
        )
    if not hmac.compare_digest(api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Clé API invalide.",
        )


class SlidingWindowRateLimiter:
    """Limiteur à fenêtre glissante en mémoire, isolé par client (IP)."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, client_id: str, max_requests: int, window_seconds: float = 60.0, now: float | None = None) -> float:
        """Enregistre un hit et renvoie 0 si la requête est acceptée.

        Si la limite est dépassée, renvoie le nombre de secondes à attendre avant
        que le plus ancien hit sorte de la fenêtre (= en-tête `Retry-After`).
        `now` est injectable pour tester sans `sleep`.
        """
        now = time.monotonic() if now is None else now
        hits = self._hits[client_id]
        while hits and hits[0] <= now - window_seconds:
            hits.popleft()
        if len(hits) >= max_requests:
            return round(hits[0] + window_seconds - now, 2)
        hits.append(now)
        return 0.0

    def reset(self) -> None:
        """Vide l'état (tests, redémarrage à chaud)."""
        self._hits.clear()


_rate_limiter = SlidingWindowRateLimiter()


def reset_rate_limiter() -> None:
    """Réinitialise le limiteur global (isolation entre tests)."""
    _rate_limiter.reset()


def rate_limit(request: Request) -> None:
    """429 si le client dépasse `settings.rate_limit_per_minute` (0 = désactivé)."""
    limit = settings.rate_limit_per_minute
    if limit <= 0:
        return
    client_id = request.client.host if request.client else "unknown"
    retry_after = _rate_limiter.check(client_id, limit)
    if retry_after > 0:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Trop de requêtes : limite de {limit}/min dépassée.",
            headers={"Retry-After": str(retry_after)},
        )
