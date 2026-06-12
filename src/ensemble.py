"""
Stage 4c — Ensemble the two unsupervised detectors into ONE production
Tax Compliance Deviation Score.

Why two models, one score:
  - Isolation Forest (scoring.py): tabular footprint-vs-declared mismatch. Wins on
    SELF-evaders — one person whose own books don't add up.
  - GraphSAGE autoencoder (gnn_detector.py): relational anomaly via message-passing
    over shared-household edges. Wins on proxy/benami PRINCIPALS who look compliant
    alone but sit in an anomalous network.

We quantile-align the GNN score onto the IF score's distribution (histogram
matching) so both live in the same 0-100 units and the audit threshold keeps its
meaning, then blend:

    combined = IF_WEIGHT * IF + (1 - IF_WEIGHT) * GNN_aligned

This keeps the IF's headline accuracy (AP) essentially intact while inheriting most
of the GNN's proxy-network recall. NO ground truth is used here (the wall holds) —
it is a pure combination of two model outputs.

Falls back to the IF score alone when the GNN output is unavailable (e.g. no torch).

Writes two columns back into data/resolved/entity_scores.csv:
  gnn_score (the relational anomaly), deviation_score_combined (the production score).
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "data", "resolved")

IF_WEIGHT = 0.7          # weight on the Isolation Forest (tabular) score; the
                         # remainder weights the quantile-aligned GNN (relational)
                         # score. 0.7 keeps IF's overall AP nearly intact while
                         # lifting proxy-principal recall; lower it toward 0.5 to
                         # trade a little AP for more relational recall.


def _quantile_align(source, reference):
    """Re-express `source` in `reference`'s distribution (histogram matching), so two
    differently-scaled anomaly scores become comparable / blendable."""
    ref_sorted = np.sort(np.asarray(reference, dtype=float))
    pct = pd.Series(source).rank(pct=True).to_numpy()
    return np.quantile(ref_sorted, np.clip(pct, 0.0, 1.0))


def combine(if_weight=IF_WEIGHT):
    iso_path = os.path.join(RES, "entity_scores.csv")
    gnn_path = os.path.join(RES, "gnn_scores.csv")
    iso = pd.read_csv(iso_path)
    # idempotent: drop any columns we are about to (re)compute
    iso = iso.drop(columns=[c for c in ("gnn_score", "deviation_score_combined")
                            if c in iso.columns])

    if not os.path.exists(gnn_path):
        print("  GNN scores not found — combined score falls back to Isolation Forest only.")
        iso["gnn_score"] = np.nan
        iso["deviation_score_combined"] = iso["deviation_score"]
        iso.to_csv(iso_path, index=False)
        return iso

    gnn = pd.read_csv(gnn_path)
    df = iso.merge(gnn, on="entity_id", how="left")
    if_score = df["deviation_score"].to_numpy()
    # any entity without a GNN score falls back to its IF score for the blend
    gnn_filled = df["gnn_score"].fillna(df["deviation_score"]).to_numpy()
    gnn_aligned = _quantile_align(gnn_filled, if_score)
    blended = if_weight * if_score + (1 - if_weight) * gnn_aligned
    # Remap the blend back onto the IF score's distribution: this preserves the
    # improved RANKING (AP / proxy recall) while keeping the IF's 0-100 scale, so
    # the audit threshold and KPIs keep their meaning (a convex blend would
    # otherwise compress scores toward the middle and shift what "60" means).
    df["deviation_score_combined"] = np.round(_quantile_align(blended, if_score), 1)

    df = df.sort_values("deviation_score_combined", ascending=False)
    df.to_csv(iso_path, index=False)
    print(f"  combined {len(df)} entities into one score (IF_WEIGHT={if_weight})")
    print(f"  combined deviation range: "
          f"{df.deviation_score_combined.min():.1f} - {df.deviation_score_combined.max():.1f}")
    return df


if __name__ == "__main__":
    combine()
