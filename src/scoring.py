"""
Stage 4 — Tax Compliance Deviation Score (Isolation Forest) + audit trail.

Unsupervised: fits an Isolation Forest on each entity's financial footprint,
expressed mainly as FOOTPRINT-TO-DECLARED ratios. Why ratios:
  - An evader (big assets, tiny declared income) -> very large ratios -> outlier.
  - A legitimately wealthy COMPLIANT filer (big assets, big declared income)
    -> normal ratios -> inlier. So we flag the *mismatch*, not just wealth.

The model never sees true income or the is_evader label (the wall holds).

Outputs data/resolved/entity_scores.csv with:
  entity_id, deviation_score (0-100), implied_income, declared_income,
  is_filer, audit_trail (JSON list of human-readable reasons).
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(ROOT, "data", "resolved")

DECL_FLOOR = 1_200_000       # imputed minimum income for someone who appears in the
                             # asset databases (~population median). A realistic floor:
                             # the old 50k gave declared==0 cases an unbounded
                             # footprint-to-declared ratio, so non-filers dominated
                             # every flag and under-reporting FILERS never surfaced.


def engineer(fdf):
    """Estimate an INDEPENDENT 'implied income' from each footprint signal, then
    combine them robustly. Each coefficient inverts a plausible economic
    relationship a tax authority would assume (consumption/assets ~ income) —
    these are domain priors, NOT fitted on ground truth.
    """
    f = fdf.copy()
    annual_elec = f["max_monthly_bill"] * 12
    decl = f["declared_income"].clip(lower=0) + DECL_FLOOR
    f["annual_electricity"] = annual_elec

    # per-signal income estimates (each ~ recovers annual income from one asset)
    inc_elec = (f["max_monthly_bill"] - 3000).clip(lower=0) * 111.0     # bill model inverse
    inc_veh = (f["max_vehicle_cc"] - 800).clip(lower=0) * 3125.0        # cc model inverse
    inc_prop = f["total_property_value"] / 4.0                          # ~property affordability
    inc_trav = f["intl_trips"] * 500_000.0                             # cost per trip proxy

    # robust combine: median of the signals the person actually has (>0),
    # which averages out per-signal noise. Fall back to max if none.
    sig = np.vstack([inc_elec, inc_veh, inc_prop, inc_trav]).astype(float).T
    sig_masked = np.where(sig > 0, sig, np.nan)
    with np.errstate(all="ignore"):
        implied = np.nanmedian(sig_masked, axis=1)
    implied = np.where(np.isnan(implied), sig.max(axis=1), implied)
    f["implied_income"] = np.maximum(implied, f["declared_income"].clip(lower=0))

    # footprint-to-declared ratios (explainable signals for the audit trail)
    f["elec_to_declared"] = annual_elec / decl
    f["property_to_declared"] = f["total_property_value"] / decl
    f["vehicle_to_declared"] = f["total_vehicle_value"] / decl
    f["travel_to_declared"] = f["intl_trips"] / (decl / 1_000_000)
    f["implied_to_declared"] = f["implied_income"] / decl

    # NETWORK signal: income implied by hidden assets held by non-filing
    # household associates (proxies), relative to declared income. This is what
    # catches a PRINCIPAL whose own footprint looks compliant.
    netcol = "network_nonfiler_asset_value"
    f[netcol] = f[netcol] if netcol in f else 0.0
    f["network_implied_income"] = f[netcol] / 4.0          # same property->income inverse
    f["network_to_declared"] = f["network_implied_income"] / decl
    return f


# Isolation Forest performs best on the mismatch RATIOS (+ filer flag): these
# isolate high-lifestyle/low-declared people while leaving legitimately wealthy
# COMPLIANT filers (high ratios cancel out) as inliers. Absolute magnitude
# features were dropped because they made IF flag merely-rich compliant people.
OWN_FEATURES = [
    "implied_to_declared", "elec_to_declared", "property_to_declared",
    "vehicle_to_declared", "travel_to_declared", "is_filer",
]
# Adding the network signal lets the SAME model also catch proxy-using principals.
MODEL_FEATURES = OWN_FEATURES + ["network_to_declared"]

# friendly labels + value formatters for the audit trail
LABELS = {
    "implied_to_declared": "overall lifestyle vs declared income",
    "elec_to_declared": "electricity bills vs declared income",
    "property_to_declared": "property holdings vs declared income",
    "vehicle_to_declared": "vehicle value vs declared income",
    "travel_to_declared": "international travel vs declared income",
    "network_to_declared": "hidden assets held by household associates vs declared income",
    "implied_income": "lifestyle-implied income",
    "intl_trips": "international trips (last year)",
    "declared_income": "declared income",
}

RATIO_FEATURES = {"implied_to_declared", "elec_to_declared", "property_to_declared",
                  "vehicle_to_declared", "travel_to_declared", "network_to_declared"}


def _pkr(x):
    x = float(x)
    if x >= 1e7:
        return f"Rs {x/1e7:.2f} crore"
    if x >= 1e5:
        return f"Rs {x/1e5:.2f} lakh"
    return f"Rs {x:,.0f}"


def build_audit_trail(row, pct, raw):
    """Model-aware, transparent reasons: which footprint signals are extreme
    (high population percentile) for this entity, in plain PKR terms."""
    reasons = []
    # rank features by how extreme they are vs the population
    ranked = sorted(
        [(f, pct[f]) for f in MODEL_FEATURES if f not in ("is_filer", "declared_income")],
        key=lambda t: t[1], reverse=True,
    )
    for f, p in ranked[:4]:
        if p < 0.80:
            continue
        if f == "network_to_declared":
            if raw.get("network_nonfiler_asset_value", 0) > 0:
                reasons.append(
                    f"linked to {int(raw.get('network_neighbor_count', 0))} non-filing "
                    f"household associate(s) holding ~{_pkr(raw['network_nonfiler_asset_value'])} "
                    f"in assets (likely proxy/benami holdings)"
                )
        elif f in RATIO_FEATURES:
            reasons.append(
                f"{LABELS[f]} in the top {100-int(p*100)}% of the population "
                f"(ratio {raw[f]:.1f}x)"
            )
        elif f == "intl_trips":
            if raw[f] > 0:
                reasons.append(f"{int(raw[f])} international trips while "
                               f"declaring {_pkr(raw['declared_income'])} income")
        elif f == "implied_income":
            reasons.append(f"{LABELS[f]} of ~{_pkr(raw[f])} "
                           f"(top {100-int(p*100)}% of population)")
    if raw["is_filer"] == 0:
        reasons.insert(0, "NON-FILER: no tax return on record despite the assets below")
    # headline gap
    reasons.append(
        f"Lifestyle implies ~{_pkr(raw['implied_income'])}/yr income, "
        f"but declared {_pkr(raw['declared_income'])}"
    )
    return reasons


def _fit_iso(f, features):
    """Fit Isolation Forest on the given features, return a 0-100 deviation score."""
    X = f[features].copy()
    for c in features:
        if c != "is_filer":
            X[c] = np.log1p(X[c].clip(lower=0))
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=42)
    iso.fit(X)
    anomaly = -iso.score_samples(X)
    dev = 100 * (anomaly - anomaly.min()) / (anomaly.max() - anomaly.min() + 1e-9)
    return dev, iso, float(anomaly.min()), float(anomaly.max())


def _save_model_bundle(f, iso, amin, amax):
    """Persist the fitted model + population reference so a NEW individual can be
    scored later (inference) without re-running the whole pipeline. See
    src/score_person.py."""
    import pickle
    keep = list(dict.fromkeys(MODEL_FEATURES + ["implied_income", "declared_income"]))
    bundle = {
        "model": iso,
        "features": MODEL_FEATURES,
        "anomaly_min": amin,
        "anomaly_max": amax,
        "population": f[keep].reset_index(drop=True),   # for percentile-based audit trail
    }
    with open(os.path.join(RES_DIR, "model_bundle.pkl"), "wb") as fh:
        pickle.dump(bundle, fh)


def score():
    fdf = pd.read_csv(os.path.join(RES_DIR, "entity_features.csv"))
    f = engineer(fdf)

    # Production score uses own + NETWORK features; the own-only score is kept
    # for the comparison that demonstrates the graph's contribution (principals).
    dev, iso_net, amin, amax = _fit_iso(f, MODEL_FEATURES)
    dev_own, _, _, _ = _fit_iso(f, OWN_FEATURES)
    _save_model_bundle(f, iso_net, amin, amax)

    # population percentile of each feature (for the audit trail)
    pct_rank = {c: f[c].rank(pct=True).values for c in MODEL_FEATURES if c != "is_filer"}

    out_rows = []
    for i, eid in enumerate(f["entity_id"].astype(int)):
        pct = {c: pct_rank[c][i] for c in pct_rank}
        raw = {c: f[c].iloc[i] for c in MODEL_FEATURES}
        raw["implied_income"] = f["implied_income"].iloc[i]
        raw["declared_income"] = f["declared_income"].iloc[i]
        raw["intl_trips"] = f["intl_trips"].iloc[i]
        raw["network_nonfiler_asset_value"] = f["network_nonfiler_asset_value"].iloc[i]
        raw["network_neighbor_count"] = f["network_neighbor_count"].iloc[i]
        trail = build_audit_trail(f.iloc[i], pct, raw)
        out_rows.append({
            "entity_id": eid,
            "deviation_score": round(float(dev[i]), 1),
            "deviation_score_own": round(float(dev_own[i]), 1),
            "implied_income": int(f["implied_income"].iloc[i]),
            "declared_income": int(f["declared_income"].iloc[i]),
            "tax_paid": int(f["tax_paid"].iloc[i]),
            "is_filer": int(f["is_filer"].iloc[i]),
            "network_nonfiler_assets": int(f["network_nonfiler_asset_value"].iloc[i]),
            "network_neighbors": int(f["network_neighbor_count"].iloc[i]),
            "audit_trail": json.dumps(trail, ensure_ascii=False),
        })

    out = pd.DataFrame(out_rows).sort_values("deviation_score", ascending=False)
    out.to_csv(os.path.join(RES_DIR, "entity_scores.csv"), index=False)

    print(f"  scored {len(out)} entities with Isolation Forest (own + network)")
    print(f"  deviation score range: {out.deviation_score.min():.1f} - {out.deviation_score.max():.1f}")
    print("\n  Top 5 flagged entities:")
    for _, r in out.head(5).iterrows():
        print(f"   #{r.entity_id}  score={r.deviation_score:>5}  "
              f"declared={_pkr(r.declared_income)}  implied~{_pkr(r.implied_income)}")
        for reason in json.loads(r.audit_trail)[:3]:
            print(f"       - {reason}")
    print("\n  wrote data/resolved/entity_scores.csv")
    return out


if __name__ == "__main__":
    score()
