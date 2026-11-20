"""Deterministic development splits derived from benchmark query IDs."""

from __future__ import annotations

import hashlib


def make_dev_split(
    query_ids: list[str] | tuple[str, ...],
    fraction: float = 0.2,
    salt: str = "paper-search-dev-v1",
) -> set[str]:
    """Select a stable development subset of query IDs.

    Query IDs are ordered by the SHA-256 digest of ``salt:query_id`` and the
    smallest ``fraction`` share is returned, so the split depends only on the
    IDs, fraction, and salt, never on input ordering or RNG state.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("Dev fraction must be in (0, 1).")
    if not query_ids:
        raise ValueError("Cannot split an empty query ID collection.")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("Query IDs must be unique.")

    def digest(query_id: str) -> str:
        return hashlib.sha256(f"{salt}:{query_id}".encode()).hexdigest()

    n_dev = max(1, round(len(query_ids) * fraction))
    ranked = sorted(query_ids, key=digest)
    return set(ranked[:n_dev])
