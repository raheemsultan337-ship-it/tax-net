"""
Score a NEW individual against the already-trained model (inference).

This is the ML-inference counterpart to scoring.py: instead of re-fitting, it
loads the frozen Isolation Forest + population reference saved by scoring.py and
predicts a Tax Compliance Deviation Score + audit trail for one new person.

    from score_person import score_person
    result = score_person({
        "declared_income": 0, "is_filer": False,
        "max_vehicle_cc": 3000, "total_vehicle_value": 25_000_000,
        "total_property_value": 60_000_000, "max_monthly_bill": 400_000,
        "intl_trips": 6,
    })
    print(result["deviation_score"], result["audit_trail"])
"""

import os
import pickle

import numpy as np
import pandas as pd

import scoring   # reuse engineer(), build_audit_trail(), MODEL_FEATURES, _pkr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(ROOT, "data", "resolved")
BUNDLE = os.path.join(RES_DIR, "model_bundle.pkl")

# raw inputs a user supplies, with safe defaults
DEFAULTS = {
    "declared_income": 0.0,
    "is_filer": 0,
    "max_vehicle_cc": 0.0,
    "total_vehicle_value": 0.0,
    "total_property_value": 0.0,
    "max_monthly_bill": 0.0,
    "intl_trips": 0.0,
    # advanced: undeclared assets held by household associates (proxy signal)
    "network_nonfiler_asset_value": 0.0,
    "network_neighbor_count": 0.0,
}


def load_bundle():
    if not os.path.exists(BUNDLE):
        raise FileNotFoundError(
            "model_bundle.pkl not found — run the pipeline first (python src/run_pipeline.py)."
        )
    with open(BUNDLE, "rb") as fh:
        return pickle.load(fh)


def _feature_row(inputs):
    """Build the one-row raw feature frame engineer() expects."""
    p = {**DEFAULTS, **{k: v for k, v in inputs.items() if v is not None}}
    p["is_filer"] = 1.0 if (p["is_filer"] or float(p["declared_income"]) > 0) else 0.0
    row = {
        "entity_id": -1,
        "max_monthly_bill": float(p["max_monthly_bill"]),
        "max_vehicle_cc": float(p["max_vehicle_cc"]),
        "total_vehicle_value": float(p["total_vehicle_value"]),
        "total_property_value": float(p["total_property_value"]),
        "intl_trips": float(p["intl_trips"]),
        "declared_income": float(p["declared_income"]),
        "tax_paid": 0.0,
        "is_filer": p["is_filer"],
        "network_nonfiler_asset_value": float(p["network_nonfiler_asset_value"]),
        "network_neighbor_count": float(p["network_neighbor_count"]),
    }
    return pd.DataFrame([row])


def score_person(inputs):
    bundle = load_bundle()
    model, feats = bundle["model"], bundle["features"]
    amin, amax, pop = bundle["anomaly_min"], bundle["anomaly_max"], bundle["population"]

    f = scoring.engineer(_feature_row(inputs))

    # same preprocessing as training, then frozen-model inference
    X = f[feats].copy()
    for c in feats:
        if c != "is_filer":
            X[c] = np.log1p(X[c].clip(lower=0))
    anomaly = -model.score_samples(X)[0]
    dev = 100 * (anomaly - amin) / (amax - amin + 1e-9)
    dev = float(np.clip(dev, 0, 100))

    # percentile of each feature vs the trained population (for the audit trail)
    pct = {}
    for c in feats:
        if c == "is_filer":
            continue
        val = float(f[c].iloc[0])
        pct[c] = float((pop[c] <= val).mean()) if c in pop else 0.0

    raw = {c: float(f[c].iloc[0]) for c in feats}
    raw["implied_income"] = float(f["implied_income"].iloc[0])
    raw["declared_income"] = float(f["declared_income"].iloc[0])
    raw["intl_trips"] = float(f["intl_trips"].iloc[0])
    raw["network_nonfiler_asset_value"] = float(f["network_nonfiler_asset_value"].iloc[0])
    raw["network_neighbor_count"] = float(f["network_neighbor_count"].iloc[0])

    trail = scoring.build_audit_trail(f.iloc[0], pct, raw)
    # where does this score sit vs the population?
    population_dev_pct = None
    return {
        "deviation_score": round(dev, 1),
        "implied_income": int(raw["implied_income"]),
        "declared_income": int(raw["declared_income"]),
        "is_filer": int(raw["is_filer"]),
        "audit_trail": trail,
    }


if __name__ == "__main__":
    # demo: a flagrant non-filer with a luxury footprint
    demo = {
        "declared_income": 0, "is_filer": False,
        "max_vehicle_cc": 3000, "total_vehicle_value": 25_000_000,
        "total_property_value": 60_000_000, "max_monthly_bill": 400_000,
        "intl_trips": 6,
    }
    r = score_person(demo)
    print(f"Deviation Score: {r['deviation_score']}/100")
    print(f"Declared: {scoring._pkr(r['declared_income'])}   "
          f"Lifestyle-implied: ~{scoring._pkr(r['implied_income'])}")
    print("Audit trail:")
    for reason in r["audit_trail"]:
        print(f"  - {reason}")
