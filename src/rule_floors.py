"""
Stage 4c.5 — Rule-based lifestyle-income floors (SECONDARY, explainable layer).

This is NOT the headline score. The production Tax Compliance Deviation Score is
the unsupervised IF + GNN ensemble (scoring.py / ensemble.py). This module adds a
fully deterministic, plain-rupee cross-check ported from the donor's rules engine:
each lifestyle signal implies a conservative income FLOOR, with the assumption
written next to it; the strongest floor is the estimated income; expected tax minus
tax paid is the gap. Used by the audit trail and the dashboard to explain WHY an
entity looks suspicious in human terms — and as an independent sanity check on the
ML ranking.

Re-implemented against the FLAT entity_features.csv (tax-net's graph stores no
per-node engine_cc / dc_value), so a couple of donor refinements are approximated:
  - vehicle floor uses the single largest engine (max_vehicle_cc), not a per-vehicle
    half-weighted sum;
  - household size is the shared-address neighbour count;
  - the 183-day non-resident damper is omitted (days-abroad isn't in the features).

Reads ONLY data/resolved/{entity_features,mentions}.csv — never ground truth.
Writes data/resolved/entity_floors.csv and data/resolved/factors.json.
"""

import os
import json

import pandas as pd

from tax_slabs import annual_tax

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(ROOT, "data", "resolved")

# Engine size -> the annual income (PKR) lifestyle plausibly requires. Floors are
# deliberately conservative: the cheapest credible owner profile.
CC_INCOME_FLOOR = [
    (2500, 15_000_000), (1600, 6_000_000), (1200, 2_500_000),
    (1000, 1_500_000), (600, 800_000), (0, 400_000),
]
POWER_CEILING_RATE = 0.06       # power spend <= 6% of gross income (a ceiling)
PROPERTY_YEARS = 8              # property bought from ~8 years of income
TRIP_INCOME_FLOOR = 1_500_000   # ~1.5M income per annual intl trip
PROPERTY_IMPLIED_DIVISOR = 4.0  # network assets -> implied income
LOW_DECLARED = 600_000

BANDS = [(70, "high"), (40, "medium"), (15, "low"), (0, "none")]


def _vehicle_floor(max_cc: float) -> int:
    for min_cc, floor in CC_INCOME_FLOOR:
        if max_cc >= min_cc:
            return floor
    return 0


def _band(score: int) -> str:
    return next(b for t, b in BANDS if score >= t)


def _names(mentions: pd.DataFrame) -> dict[int, tuple[str, str]]:
    """Representative display name + city per entity (first non-empty)."""
    out: dict[int, tuple[str, str]] = {}
    for eid, grp in mentions.groupby("entity_id"):
        name = next((n for n in grp["norm_name"].astype(str) if n and n != "nan"), "")
        city = next((c for c in grp.get("city", pd.Series([], dtype=str)).astype(str)
                     if c and c != "nan"), "")
        out[int(eid)] = (name.title(), city.title())
    return out


def compute_floors(data_dir: str | None = None) -> dict:
    res = os.path.join(data_dir, "resolved") if data_dir else RES_DIR
    feats = pd.read_csv(os.path.join(res, "entity_features.csv"))
    try:
        mentions = pd.read_csv(os.path.join(res, "mentions.csv"))
        names = _names(mentions)
    except FileNotFoundError:
        names = {}

    rows = []
    factors_out: dict[str, dict] = {}
    for r in feats.itertuples():
        eid = int(r.entity_id)
        factors = []

        bill = float(r.max_monthly_bill)
        if bill > 0:
            hh = min(int(r.shared_address_neighbors) + 1, 3)
            implied = int(bill * 12 / POWER_CEILING_RATE / hh)
            hh_note = f", shared across a household of {hh}" if hh > 1 else ""
            factors.append({
                "factor": "electricity", "implied_income": implied,
                "detail": f"avg bill {int(bill):,} PKR/month; conservative floor "
                          f"assuming power takes up to {POWER_CEILING_RATE:.0%} of "
                          f"gross income{hh_note}",
            })
        max_cc = float(r.max_vehicle_cc)
        if max_cc > 0:
            implied = _vehicle_floor(max_cc)
            n = int(r.n_vehicles)
            factors.append({
                "factor": "vehicles", "implied_income": implied,
                "detail": f"{n} vehicle(s), largest {int(max_cc)}cc; "
                          f"engine-size income floor",
            })
        prop_val = float(r.total_property_value)
        if prop_val > 0:
            implied = int(prop_val / PROPERTY_YEARS)
            factors.append({
                "factor": "property", "implied_income": implied,
                "detail": f"{int(r.n_properties)} holding(s), total value "
                          f"{int(prop_val):,} PKR; assumes purchase from "
                          f"~{PROPERTY_YEARS} years of income",
            })
        trips = int(r.intl_trips)
        if trips > 0:
            implied = trips * TRIP_INCOME_FLOOR
            factors.append({
                "factor": "travel", "implied_income": implied,
                "detail": f"{trips} intl trip(s); ~{TRIP_INCOME_FLOOR:,} PKR "
                          f"income per annual trip",
            })

        estimated = max((f["implied_income"] for f in factors), default=0)
        declared = float(r.declared_income)
        is_filer = int(r.is_filer) == 1
        tax_paid = float(r.tax_paid)
        expected_tax = annual_tax(estimated)
        tax_gap = max(0.0, expected_tax - tax_paid)
        gap_ratio = tax_gap / expected_tax if expected_tax else 0.0
        significance = tax_gap / (tax_gap + 300_000)
        score = int(round(100 * significance * (0.5 + 0.5 * gap_ratio)))

        notes = []
        if not is_filer and expected_tax > 0:
            notes.append("no tax return on file (ghost) despite a taxable lifestyle")
        net_hidden = float(getattr(r, "network_nonfiler_asset_value", 0.0) or 0.0)
        if net_hidden > 2_000_000:
            n_nb = int(getattr(r, "network_neighbor_count", 0) or 0)
            implied_net = int(net_hidden / PROPERTY_IMPLIED_DIVISOR)
            notes.append(
                f"linked to {n_nb} non-filing household associate(s) holding "
                f"~{int(net_hidden):,} PKR in assets (implied income "
                f"~{implied_net:,} PKR) — possible proxy/benami holding")

        filer_status = "filer" if (is_filer and declared > 0) else "absent"
        name, city = names.get(eid, ("", ""))
        rows.append({
            "entity_id": eid,
            "display_name": name or f"Entity {eid}",
            "city": city,
            "estimated_income_pkr": int(estimated),
            "declared_income_pkr": int(declared),
            "filer_status": filer_status,
            "tax_paid_pkr": int(tax_paid),
            "expected_tax_pkr": int(expected_tax),
            "tax_gap_pkr": int(tax_gap),
            "rule_deviation_score": score,
            "band": _band(score),
        })
        factors_out[str(eid)] = {"factors": factors, "notes": notes}

    df = pd.DataFrame(rows).sort_values("rule_deviation_score", ascending=False)
    os.makedirs(res, exist_ok=True)
    df.to_csv(os.path.join(res, "entity_floors.csv"), index=False)
    with open(os.path.join(res, "factors.json"), "w", encoding="utf-8") as fh:
        json.dump(factors_out, fh, indent=1, ensure_ascii=False)

    stats = {
        "entities_scored": len(df),
        "bands": df.band.value_counts().to_dict(),
        "total_tax_gap_pkr": int(df.tax_gap_pkr.sum()),
        "recoverable_from_high_band_pkr": int(df[df.band == "high"].tax_gap_pkr.sum()),
    }
    print(f"  rule floors: {len(df)} entities; bands {stats['bands']}")
    print(f"  total lifestyle-implied tax gap: Rs {stats['total_tax_gap_pkr']:,}")
    print("  wrote data/resolved/entity_floors.csv + factors.json")
    return stats


if __name__ == "__main__":
    compute_floors()
