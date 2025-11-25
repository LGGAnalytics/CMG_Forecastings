from __future__ import annotations

import pandas as pd


def assign_store_models(
    df: pd.DataFrame,
    cohort_col: str = "cohort",
    pooled_col: str = "WAPE_model_pct",
    peer_col: str = "WAPE_peer_pct",
    assignment_col: str = "assigned_model",
    threshold_pp: float = 3.0,
) -> pd.DataFrame:
    """
    Add a per-store assignment column ("pooled" or "peer") following the business rules:

    - Long-history stores: always "pooled" (no comparisons, no tie-breaks).
    - Short-history stores: default to "peer" and only switch to "pooled" if
      pooled WAPE is better by at least `threshold_pp` percentage points.

      Define improvement = pooled_WAPE - peer_WAPE (in percentage points).
      If improvement <= -threshold_pp -> pooled is better by >= threshold -> choose "pooled".
      Else -> keep "peer".

    - If peer WAPE is missing for a short store, still keep "peer" by default.
    """

    if cohort_col not in df.columns:
        raise KeyError(f"Missing required column: {cohort_col}")
    if pooled_col not in df.columns:
        raise KeyError(f"Missing required column: {pooled_col}")
    if peer_col not in df.columns:
        # Peer may be empty for some workflows, but the column should exist (possibly all NaN)
        raise KeyError(f"Missing required column: {peer_col}")

    out = df.copy()
    cohort_norm = out[cohort_col].astype(str).str.strip().str.lower()
    pooled = pd.to_numeric(out[pooled_col], errors="coerce")
    peer = pd.to_numeric(out[peer_col], errors="coerce")

    assigned = pd.Series("peer", index=out.index, dtype=object)
    assigned.loc[cohort_norm == "long"] = "pooled"

    improvement = pooled - peer
    switch_mask = (cohort_norm == "short") & improvement.le(-float(threshold_pp))
    assigned.loc[switch_mask] = "pooled"

    out[assignment_col] = assigned.values
    return out


__all__ = ["assign_store_models"]
