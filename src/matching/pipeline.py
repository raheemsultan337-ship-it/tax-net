"""Tier 1 entity resolution pipeline.

Reads  data/observable/*.csv
Writes data/er_output/clusters.csv         record_id -> entity_id
       data/er_output/match_pairs.csv      accepted edges with evidence
       data/er_output/borderline_pairs.csv undecided pairs for Tier 2 / Tier 4
       data/er_output/er_meta.json         learned weights, thresholds, stats
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd

from .blocking import candidate_pairs
from .cluster import build_clusters
from .feedback import load_feedback
from .fellegi import FellegiSunter
from .records import Record, load_records

MATCH_PROB = 0.95
BORDERLINE_PROB = 0.20

# Tier 1 may auto-merge only when a hard identifier agrees. Pairs whose
# evidence is purely name/father/city/DOB similarity - however strong - are
# escalated to the smarter tiers instead. Names alone never merge people.
HARD_EVIDENCE = {"cnic_exact", "cnic_close", "phone_match",
                 "addr_exact", "addr_house_match"}


def run(data_dir: str = "data", seed: int = 7) -> dict:
    t0 = time.time()
    rng = random.Random(seed)
    records = load_records(data_dir)
    pairs, block_stats = candidate_pairs(records)

    # Fold any auditor decisions from previous sessions back into calibration.
    idx_of = {r.record_id: i for i, r in enumerate(records)}
    extra_matches, extra_nonmatches = [], []
    for rec_a, rec_b, label in load_feedback(data_dir):
        if rec_a in idx_of and rec_b in idx_of:
            pair = (idx_of[rec_a], idx_of[rec_b])
            (extra_matches if label == "same" else extra_nonmatches).append(pair)

    model = FellegiSunter()
    fit_stats = model.fit(records, candidate_count=len(pairs), rng=rng,
                          extra_matches=extra_matches,
                          extra_nonmatches=extra_nonmatches)

    matches: list[dict] = []
    borderline: list[dict] = []
    match_idx: list[tuple[int, int]] = []
    for i, j in pairs:
        score, bands = model.score(records[i], records[j])
        prob = model.posterior(score)
        # Father/son veto: a conflicting father name means a different person
        # unless the CNIC itself says otherwise. Same name + same house +
        # different father is the classic father/son signature.
        father_veto = ("father_diff" in bands
                       and "cnic_exact" not in bands
                       and "cnic_close" not in bands)
        # DOB veto: a confirmed birth-date conflict overrides everything short
        # of a matching CNIC (guards twins / same-name relatives).
        dob_veto = ("dob_diff" in bands
                    and "cnic_exact" not in bands
                    and "cnic_close" not in bands)
        if prob >= MATCH_PROB and not father_veto and not dob_veto \
                and any(b in HARD_EVIDENCE for b in bands):
            match_idx.append((i, j))
            matches.append(_pair_row(records[i], records[j], score, prob, bands, "tier1"))
        elif prob >= BORDERLINE_PROB:
            borderline.append(_pair_row(records[i], records[j], score, prob, bands, ""))

    clusters = build_clusters(len(records), match_idx)
    cluster_rows = []
    for k, group in enumerate(clusters):
        entity_id = f"E{k + 1:06d}"
        for idx in group:
            r = records[idx]
            cluster_rows.append({
                "record_id": r.record_id,
                "registry": r.registry,
                "entity_id": entity_id,
                "cluster_size": len(group),
            })

    out_dir = Path(data_dir) / "er_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cluster_rows).to_csv(out_dir / "clusters.csv", index=False,
                                      encoding="utf-8-sig")
    pd.DataFrame(matches).to_csv(out_dir / "match_pairs.csv", index=False,
                                 encoding="utf-8-sig")
    pd.DataFrame(borderline).to_csv(out_dir / "borderline_pairs.csv", index=False,
                                    encoding="utf-8-sig")

    meta = {
        "n_records": len(records),
        "blocking": block_stats,
        "fellegi_sunter": fit_stats,
        "thresholds": {"match_prob": MATCH_PROB, "borderline_prob": BORDERLINE_PROB},
        "n_match_pairs": len(matches),
        "n_borderline_pairs": len(borderline),
        "n_entities": len(clusters),
        "n_multi_record_entities": sum(1 for g in clusters if len(g) > 1),
        "runtime_sec": round(time.time() - t0, 1),
    }
    (out_dir / "er_meta.json").write_text(json.dumps(meta, indent=2),
                                          encoding="utf-8")
    return meta


def _pair_row(a: Record, b: Record, score: float, prob: float,
              bands: list[str], tier: str) -> dict:
    return {
        "record_a": a.record_id, "registry_a": a.registry, "name_a": a.raw_name,
        "record_b": b.record_id, "registry_b": b.registry, "name_b": b.raw_name,
        "score_bits": round(score, 2),
        "posterior_prob": round(prob, 4),
        "evidence": "|".join(bands),
        "decided_by": tier,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    meta = run(args.data_dir)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
