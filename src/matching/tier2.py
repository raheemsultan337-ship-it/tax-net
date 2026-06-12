"""Tier 2: multilingual embedding similarity for borderline pairs.

A sentence-transformer trained on 50+ languages embeds names from Urdu
script and Roman transliteration into one vector space, so محمد آصف خان and
"Mohd Asif Khan" land close together with no dictionary. The cosine bands
are calibrated unsupervised exactly like Tier 1's features: m-probabilities
from CNIC-anchored pairs, u-probabilities from random pairs.

Embedding similarity never decides alone: same-name strangers embed
identically, so promotion requires corroborating evidence (father match,
masked CNIC, address overlap, shared phone) and no contradicting hard
identifier. Runs fully offline from the project-local models/ cache.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .features import compare
from .fellegi import FellegiSunter, anchor_pairs, random_nonmatch_pairs
from .records import load_records

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
PROMOTE_PROB = 0.97
VETO = {"cnic_diff", "father_diff", "city_diff", "dob_diff"}
SUPPORT = {"father_match", "cnic_masked_match", "addr_exact",
           "addr_house_match", "addr_area_match", "phone_match", "dob_match"}
HARD = {"addr_exact", "addr_house_match", "phone_match"}
CROSS_SCRIPT_COS = 0.65
EMB_BANDS = [(0.85, "emb_high"), (0.70, "emb_mid"), (0.50, "emb_low"),
             (-1.0, "emb_diff")]
NAME_BANDS = {"name_exact", "name_close", "name_weak", "name_diff"}


def load_model(cache: str = "models"):
    from sentence_transformers import SentenceTransformer
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        return SentenceTransformer(MODEL_NAME, cache_folder=cache)
    except Exception:
        os.environ.pop("HF_HUB_OFFLINE", None)
        return SentenceTransformer(MODEL_NAME, cache_folder=cache)


def _band(cos: float) -> str:
    return next(b for t, b in EMB_BANDS if cos >= t)


def run(data_dir: str = "data", model_cache: str = "models") -> dict:
    data = Path(data_dir)
    out_dir = data / "er_output"
    records = load_records(data_dir)
    idx_of = {r.record_id: i for i, r in enumerate(records)}

    clusters = pd.read_csv(out_dir / "clusters.csv", dtype=str)
    entity_of = dict(zip(clusters.record_id, clusters.entity_id))
    borderline = pd.read_csv(out_dir / "borderline_pairs.csv", dtype=str)
    if borderline.empty:
        return {"promoted_pairs": 0, "n_entities_after": clusters.entity_id.nunique(),
                "borderline_remaining": 0, "note": "no borderline pairs"}

    meta = json.loads((out_dir / "er_meta.json").read_text(encoding="utf-8"))
    model_fs = FellegiSunter()
    model_fs.weights = meta["fellegi_sunter"]["weights"]
    model_fs.prior_logodds = meta["fellegi_sunter"]["prior_logodds"]

    rng = random.Random(11)
    anchors = anchor_pairs(records, cap=1500, rng=rng)
    randoms = random_nonmatch_pairs(records, n=4000, rng=rng)

    need = set()
    for i, j in anchors + randoms:
        need.add(records[i].raw_name)
        need.add(records[j].raw_name)
    for row in borderline.itertuples():
        need.add(records[idx_of[row.record_a]].raw_name)
        need.add(records[idx_of[row.record_b]].raw_name)
    names = sorted(need)

    st = load_model(model_cache)
    vecs = st.encode(names, normalize_embeddings=True, batch_size=256,
                     show_progress_bar=False)
    vec_of = {n: v for n, v in zip(names, vecs)}

    def cos(a: str, b: str) -> float:
        return float(vec_of[a] @ vec_of[b])

    # Unsupervised calibration of embedding bands.
    m_counts: dict[str, int] = defaultdict(int)
    for i, j in anchors:
        m_counts[_band(cos(records[i].raw_name, records[j].raw_name))] += 1
    u_counts: dict[str, int] = defaultdict(int)
    for i, j in randoms:
        u_counts[_band(cos(records[i].raw_name, records[j].raw_name))] += 1
    emb_weights = {}
    for _, band in EMB_BANDS:
        m = (m_counts[band] + 0.5) / (len(anchors) + 1)
        u = (u_counts[band] + 0.5) / (len(randoms) + 1)
        emb_weights[band] = max(-8.0, min(8.0, math.log2(m / u)))

    # Promotion pass.
    promotions = []
    merges = []
    keep_rows = []
    for row in borderline.to_dict("records"):
        ra = records[idx_of[row["record_a"]]]
        rb = records[idx_of[row["record_b"]]]
        ea, eb = entity_of[ra.record_id], entity_of[rb.record_id]
        if ea == eb:
            continue
        bands = compare(ra, rb)
        band_set = set(bands)
        promoted = False
        route = None
        prob = 0.0
        if not (band_set & VETO):
            c = cos(ra.raw_name, rb.raw_name)
            emb_band = _band(c)
            name_band = next((b for b in bands if b in NAME_BANDS), None)

            # Route A - cross-script rescue: a hard identifier agrees and the
            # embedding vouches for a name the skeleton couldn't read.
            if (band_set & HARD) and c >= CROSS_SCRIPT_COS \
                    and name_band in ("name_weak", "name_diff", None):
                route, prob = "tier2_cross_script", 1.0

            # Route B - weak-evidence accumulation: two independent
            # corroborations; embedding REPLACES the name band (they are
            # correlated views of the same evidence, never additive).
            elif len(band_set & SUPPORT) >= 2:
                name_w = model_fs.weights.get(name_band, 0.0) if name_band else 0.0
                base = sum(model_fs.weights.get(b, 0.0) for b in bands
                           if b not in NAME_BANDS)
                score = base + max(name_w, emb_weights[emb_band])
                p = model_fs.posterior(score)
                if p >= PROMOTE_PROB:
                    route, prob = "tier2_accumulation", p

            if route:
                merges.append((ea, eb))
                promotions.append({
                    "record_a": ra.record_id, "record_b": rb.record_id,
                    "name_a": ra.raw_name, "name_b": rb.raw_name,
                    "evidence": "|".join(bands), "embedding_band": emb_band,
                    "embedding_cos": round(c, 3),
                    "posterior_prob": round(prob, 4),
                    "decided_by": route,
                })
                promoted = True
        if not promoted:
            keep_rows.append(row)

    parent: dict[str, str] = {}

    def find(e: str) -> str:
        parent.setdefault(e, e)
        while parent[e] != e:
            parent[e] = parent[parent[e]]
            e = parent[e]
        return e

    for ea, eb in merges:
        ra_, rb_ = find(ea), find(eb)
        if ra_ != rb_:
            parent[rb_] = ra_
    entity_of = {rid: find(eid) for rid, eid in entity_of.items()}

    members = defaultdict(list)
    for rid, eid in entity_of.items():
        members[eid].append(rid)
    rows = [{"record_id": rid,
             "registry": records[idx_of[rid]].registry,
             "entity_id": eid,
             "cluster_size": len(rids)}
            for eid, rids in members.items() for rid in rids]
    pd.DataFrame(rows).to_csv(out_dir / "clusters.csv", index=False,
                              encoding="utf-8-sig")
    pd.DataFrame(promotions).to_csv(out_dir / "tier2_promotions.csv",
                                    index=False, encoding="utf-8-sig")
    pd.DataFrame(keep_rows).to_csv(out_dir / "borderline_pairs.csv",
                                   index=False, encoding="utf-8-sig")

    stats = {
        "device": str(st.device),
        "names_embedded": len(names),
        "embedding_band_weights": {k: round(v, 3) for k, v in emb_weights.items()},
        "promoted_pairs": len(promotions),
        "n_entities_after": len(members),
        "borderline_remaining": len(keep_rows),
    }
    meta["tier2"] = stats
    (out_dir / "er_meta.json").write_text(json.dumps(meta, indent=2),
                                          encoding="utf-8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--model-cache", default="models")
    args = ap.parse_args()
    print(json.dumps(run(args.data_dir, args.model_cache), indent=2))


if __name__ == "__main__":
    main()
