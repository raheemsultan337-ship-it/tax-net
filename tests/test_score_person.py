"""Validation for the live 'Score a new individual' inference path
(`src/score_person.py`).

These tests certify that the dashboard's interactive scoring is HONEST:

  1. It reproduces the batch pipeline's score exactly for the same person, i.e.
     it really runs the frozen Isolation Forest the rest of the pipeline trained
     (no retraining, no drift, no accidental ground-truth leakage).
  2. The score reflects the declared-vs-footprint MISMATCH, not raw wealth — a
     compliant rich filer scores far below an evader with the identical footprint.
  3. It is bounded [0, 100] and robust to empty / extreme input.
  4. Every flag carries a non-empty audit trail (explainability).
"""
import os

import numpy as np
import pandas as pd

from score_person import score_person, DEFAULTS
from conftest import RES


# raw footprint columns score_person consumes, as named in entity_features.csv
_RAW_COLS = [
    "declared_income", "is_filer", "max_vehicle_cc", "total_vehicle_value",
    "total_property_value", "max_monthly_bill", "intl_trips",
    "network_nonfiler_asset_value", "network_neighbor_count",
]


def _inputs_from_feature_row(row):
    return {c: row[c] for c in _RAW_COLS}


# ---- 1. Inference == batch pipeline (the core honesty check) -------------
def test_inference_reproduces_batch_score(features, scores):
    """For real entities spanning the whole score range, the live inference
    path must reproduce entity_scores.deviation_score (the IF own+network score).
    This proves the dashboard scores people with the SAME frozen model — it is
    not a separate, re-fit, or label-aware code path."""
    sc = scores.set_index("entity_id")
    ids = sc.sort_values("deviation_score").index
    sample = list(ids[:: max(1, len(ids) // 25)])      # ~25 entities, low→high
    assert len(sample) >= 10

    max_diff = 0.0
    for eid in sample:
        row = features[features.entity_id == eid].iloc[0]
        live = score_person(_inputs_from_feature_row(row))["deviation_score"]
        batch = float(sc.loc[eid, "deviation_score"])
        max_diff = max(max_diff, abs(live - batch))
    # both round to 1 dp from the identical model+normalisation → essentially exact
    assert max_diff <= 0.2, f"inference drifted from the trained model: {max_diff}"


# ---- 2. Honest to the ratio, not to wealth -------------------------------
def test_score_tracks_mismatch_not_wealth():
    """Two people with an IDENTICAL luxury footprint: one declares nothing, one
    declares income that matches it. The evader must score high and the compliant
    filer low — the score measures under-reporting, not how rich someone is."""
    footprint = dict(max_vehicle_cc=3000, total_vehicle_value=25_000_000,
                     total_property_value=60_000_000, max_monthly_bill=400_000,
                     intl_trips=6)
    evader = score_person({**footprint, "declared_income": 0})["deviation_score"]
    compliant = score_person({**footprint, "declared_income": 40_000_000})["deviation_score"]

    assert evader >= 70, f"flagrant non-filer not flagged: {evader}"
    assert compliant <= 45, f"compliant rich filer wrongly flagged: {compliant}"
    assert evader - compliant >= 30, \
        f"score failed to separate evasion from wealth (evader={evader}, compliant={compliant})"


def test_network_signal_raises_score():
    """A person with no personal footprint but large undeclared assets held by
    non-filing household associates (the proxy/benami principal) must score higher
    than the same person with no such network — the graph signal carries through
    to live inference."""
    base = dict(declared_income=0)
    alone = score_person(base)["deviation_score"]
    with_proxy = score_person({**base, "network_nonfiler_asset_value": 80_000_000,
                               "network_neighbor_count": 3})["deviation_score"]
    assert with_proxy > alone, \
        f"network/proxy signal ignored in inference (alone={alone}, proxy={with_proxy})"


# ---- 3. Bounded & robust -------------------------------------------------
def test_score_bounded_and_robust():
    for inp in ({}, DEFAULTS,
                dict(declared_income=0, max_vehicle_cc=6000,
                     total_property_value=5_000_000_000, max_monthly_bill=5_000_000,
                     intl_trips=50),
                dict(declared_income=10**12)):
        r = score_person(inp)
        assert 0.0 <= r["deviation_score"] <= 100.0
        assert isinstance(r["audit_trail"], list)


def test_filing_status_derived_from_declared():
    """Filing status is inferred from declared income (declared > 0 ⇒ filer),
    matching the dashboard form's contract."""
    assert score_person({"declared_income": 0})["is_filer"] == 0
    assert score_person({"declared_income": 2_000_000})["is_filer"] == 1


# ---- 4. Explainability ---------------------------------------------------
def test_flagged_individual_has_audit_trail():
    r = score_person(dict(declared_income=0, max_vehicle_cc=3000,
                          total_vehicle_value=25_000_000, total_property_value=60_000_000,
                          max_monthly_bill=400_000, intl_trips=6))
    assert r["deviation_score"] >= 70
    assert len(r["audit_trail"]) >= 1
    # the headline reason for a zero-return luxury footprint is non-filing
    assert any("NON-FILER" in reason.upper() for reason in r["audit_trail"])
