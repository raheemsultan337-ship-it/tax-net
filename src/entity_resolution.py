"""
Stage 2 — Entity resolution (thin orchestrator over the `matching` cascade).

Runs the ported multi-tier cascade and converts its output into tax-net's
canonical contract:

  Tier 1  Fellegi-Sunter probabilistic matching   (matching.pipeline)
  Tier 2  multilingual-embedding rescue (optional) (matching.tier2)
  Tier 4  collective graph resolution               (matching.tier4)

Writes:
  data/resolved/mentions.csv         record_id, source, raw_name, norm_name,
                                     dob, cnic, city, entity_id (contiguous int)
  data/resolved/match_evidence.csv   per-link evidence (for the audit trail)

The cascade reads ONLY data/observable/*.csv (+ its own data/er_output/
artifacts). It never reads the sealed answer key — the wall holds.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RES_DIR = os.path.join(DATA_DIR, "resolved")
ER_DIR = os.path.join(DATA_DIR, "er_output")
MODELS_DIR = os.path.join(ROOT, "models")


def _read_er(name):
    """Read an er_output CSV (utf-8-sig handles the BOM), tolerating absence."""
    path = os.path.join(ER_DIR, name)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _emit_match_evidence():
    """Flatten the per-tier pair tables into one evidence table for auditing."""
    frames = []
    for fname, tier in [("match_pairs.csv", "tier1"),
                        ("tier2_promotions.csv", "tier2"),
                        ("tier4_promotions.csv", "tier4")]:
        df = _read_er(fname)
        if df.empty:
            continue
        out = pd.DataFrame({
            "record_a": df.get("record_a"),
            "record_b": df.get("record_b"),
            "decided_by": df.get("decided_by", tier),
            "evidence": df.get("evidence", ""),
            "posterior": df.get("posterior_prob", ""),
        })
        frames.append(out)
    evidence = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["record_a", "record_b", "decided_by", "evidence", "posterior"])
    os.makedirs(RES_DIR, exist_ok=True)
    evidence.to_csv(os.path.join(RES_DIR, "match_evidence.csv"), index=False)
    return evidence


def _emit_mentions():
    from matching.records import load_records

    records = load_records(DATA_DIR)
    rec_by_id = {r.record_id: r for r in records}

    clusters = _read_er("clusters.csv")
    if clusters.empty:
        raise RuntimeError("ER cascade produced no clusters.csv")

    # Remap string entity ids (incl. split suffixes E000123.S1 / .F2) onto a
    # contiguous integer range — build_graph indexes entities as ints.
    uniq = sorted(clusters["entity_id"].unique())
    id_map = {e: i for i, e in enumerate(uniq)}

    rows = []
    for rec_id, eid in zip(clusters["record_id"], clusters["entity_id"]):
        r = rec_by_id[rec_id]
        rows.append({
            "record_id": rec_id,
            "source": r.registry,
            "raw_name": r.raw_name,
            "norm_name": " ".join(r.name_toks) if r.name_toks else r.raw_name,
            "dob": r.dob or "",
            "cnic": r.cnic["digits"] or "",
            "city": r.city or "",
            "entity_id": id_map[eid],
        })
    mentions = pd.DataFrame(rows)
    os.makedirs(RES_DIR, exist_ok=True)
    mentions.to_csv(os.path.join(RES_DIR, "mentions.csv"), index=False)
    _emit_match_evidence()
    return mentions


def resolve():
    from matching import pipeline, tier2, tier4

    m1 = pipeline.run(data_dir=DATA_DIR)
    print(f"  [tier1] {m1['n_entities']} entities from {m1['n_records']} records "
          f"({m1['blocking']['candidate_pairs']} candidate pairs)")
    try:
        s2 = tier2.run(data_dir=DATA_DIR, model_cache=MODELS_DIR)
        print(f"  [tier2] embeddings promoted {s2.get('promoted_pairs', 0)} pairs "
              f"(device={s2.get('device', 'n/a')})")
    except Exception as e:  # embeddings optional — degrade like the torch guard
        print(f"  [tier2] embeddings unavailable ({type(e).__name__}: {e}); "
              f"continuing with tier1+tier4")
    s4 = tier4.run(data_dir=DATA_DIR)
    print(f"  [tier4] {s4['promoted_pairs']} graph-promotions, "
          f"{s4['entities_split']} splits -> {s4['n_entities_after']} entities")

    mentions = _emit_mentions()
    print(f"  wrote data/resolved/mentions.csv ({mentions.entity_id.nunique()} entities, "
          f"{len(mentions)} records)")
    return mentions


if __name__ == "__main__":
    resolve()
