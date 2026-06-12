"""
Stage 1 — Synthetic Pakistani civic data generator (thin entrypoint).

The heavy lifting lives in the ``datagen`` package (ported from the donor and
adapted to tax-net's schema). This module wires it to tax-net's file layout and
keeps the SEED / N_PERSONS / main() contract the rest of the pipeline imports.

DESIGN PRINCIPLE (the wall):
  - Each person's OBSERVABLE footprint (vehicles, property, bills, travel) flows
    from their income via a behavioural model + rendering noise.
  - Under-reporting (the latent is_evader / role labels) is generated
    INDEPENDENTLY of how the detector scores people — no circularity.
  - The detector reads ONLY data/observable/*.csv. Ground truth (true income,
    is_evader, role, record->person linkage) goes to data/ground_truth/ and is
    opened ONLY by evaluate.py, AFTER detection.

Richness over the previous generator: per-registry Urdu script, masked/typo'd
CNICs, phone identifiers, address rendering variation, real vehicle makes, and
hand-seeded edge personas — all of which exercise the entity-resolution cascade.
"""

import os
import random

import pandas as pd

from datagen import registries as reg
from datagen.personas import generate_population, is_evader

SEED = 42
N_PERSONS = 50000

# Per-registry probability that a name is rendered in Urdu script (passport and
# FBR records stay Roman — official systems use the Roman transliteration).
URDU_NAME_PROB = {
    "real_estate": 0.15,
    "utilities": 0.06,
    "vehicles": 0.04,
}
TYPO_PROB = 0.04
CNIC_TYPO_PROB = 0.02

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBS_DIR = os.path.join(ROOT, "data", "observable")
GT_DIR = os.path.join(ROOT, "data", "ground_truth")


def _asset_implied(p) -> int:
    """Rough income the held assets imply (for the proxy's GT asset_income)."""
    v = sum(dc for dc, _ in p.properties) / 4.0
    v += sum(cc for _, cc, _ in p.vehicles) * 3000
    v += p.avg_monthly_bill * 12 / 0.04
    return int(v)


def _persons_frame(persons) -> pd.DataFrame:
    rows = []
    for p in persons:
        actual = p.actual_income
        ratio = round(p.declared_income / actual, 3) if actual > 0 else 0.0
        rows.append({
            "person_id": p.person_id,
            "gender": p.gender,
            "canonical_first": p.name_tokens[0],
            "canonical_last": p.name_tokens[-1],
            "father_first": p.father_tokens[0] if p.father_tokens else "",
            "dob": p.dob,
            "cnic": p.cnic,
            "city": p.address.city,
            "area": p.address.area,
            "household": p.address.canonical(),
            "true_income": round(actual),
            "asset_income": round(actual) if p.role != "proxy" else _asset_implied(p),
            "report_ratio": ratio,
            "declared_income": round(p.declared_income),
            "non_filer": p.filer_status == "absent",
            "is_evader": is_evader(p),
            "role": p.role,
            "principal_id": p.principal_id or "-1",
            "uses_proxy": p.role == "principal",
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs(OBS_DIR, exist_ok=True)
    os.makedirs(GT_DIR, exist_ok=True)

    rng = random.Random(SEED)
    persons = generate_population(N_PERSONS, rng)

    links: list[dict] = []
    datasets = {
        "vehicles": reg.build_vehicles(persons, rng, links, URDU_NAME_PROB["vehicles"],
                                       TYPO_PROB, CNIC_TYPO_PROB),
        "real_estate": reg.build_real_estate(persons, rng, links, URDU_NAME_PROB["real_estate"],
                                             TYPO_PROB, CNIC_TYPO_PROB),
        "utilities": reg.build_utilities(persons, rng, links, URDU_NAME_PROB["utilities"],
                                         TYPO_PROB, CNIC_TYPO_PROB),
        "travel": reg.build_travel(persons, rng, links, TYPO_PROB, CNIC_TYPO_PROB),
        "tax_returns": reg.build_tax_returns(persons, rng, links, CNIC_TYPO_PROB),
    }

    for name, df in datasets.items():
        # shuffle so record order leaks nothing about the population order
        df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        df.to_csv(os.path.join(OBS_DIR, f"{name}.csv"), index=False, encoding="utf-8")
        print(f"  observable/{name}.csv  ->  {len(df):>6} records")

    persons_df = _persons_frame(persons)
    persons_df.to_csv(os.path.join(GT_DIR, "persons.csv"), index=False, encoding="utf-8")
    pd.DataFrame(links).to_csv(os.path.join(GT_DIR, "record_linkage.csv"),
                               index=False, encoding="utf-8")

    n_ev = int(persons_df["is_evader"].sum())
    n_pr = int((persons_df["role"] == "principal").sum())
    n_px = int((persons_df["role"] == "proxy").sum())
    print(f"\n  ground_truth/persons.csv         ->  {len(persons_df)} persons")
    print(f"  ground_truth/record_linkage.csv  ->  {len(links)} record->person links")
    print(f"  planted evaders (is_evader)      ->  {n_ev} ({n_ev/len(persons_df)*100:.1f}%)")
    print(f"  proxy principals / proxies       ->  {n_pr} / {n_px}")
    print("\nDone. Detector reads ONLY data/observable/*.csv")


if __name__ == "__main__":
    main()
