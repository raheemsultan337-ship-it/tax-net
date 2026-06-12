"""
Live entity matching for the dashboard.

Takes a raw record typed by a user (name in any script, optional father / address
/ city / CNIC / phone / DOB), blocks it against the already-resolved population,
and scores it with the frozen Fellegi-Sunter weights — returning the best-matching
resolved entities with their evidence bands and posterior probabilities.

This is the cascade's Tier-1 scorer applied to one ad-hoc query (no LLM). It lets
a judge invent an input on the spot and watch the matcher reason about it live.
Reads ONLY data/observable, data/er_output, data/resolved — never ground truth.
"""

import os
import json
from collections import defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")


def load_index(data_dir: str = DATA_DIR) -> dict:
    """Load records, the fitted matcher, the record->entity map, and a block index."""
    from matching.records import load_records
    from matching.blocking import block_keys
    from matching.fellegi import FellegiSunter

    records = load_records(data_dir)
    meta = json.loads(open(os.path.join(data_dir, "er_output", "er_meta.json"),
                           encoding="utf-8").read())
    model = FellegiSunter()
    model.weights = meta["fellegi_sunter"]["weights"]
    model.prior_logodds = meta["fellegi_sunter"]["prior_logodds"]

    mentions = pd.read_csv(os.path.join(data_dir, "resolved", "mentions.csv"))
    rec2int = dict(zip(mentions.record_id.astype(str), mentions.entity_id.astype(int)))

    block_index: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        for key in block_keys(r):
            block_index[key].append(i)

    return {"records": records, "model": model, "rec2int": rec2int,
            "block_index": block_index}


def _build_query(query: dict):
    from matching.records import Record
    from matching.normalize import (
        is_urdu, name_skeletons, name_tokens, normalize_address,
        normalize_cnic, normalize_phone,
    )
    name = str(query.get("name", "") or "")
    father = str(query.get("father", "") or "")
    address = str(query.get("address", "") or "")
    return Record(
        record_id="QUERY", registry="query", raw_name=name,
        name_toks=name_tokens(name) if not is_urdu(name) else [],
        name_skels=name_skeletons(name),
        urdu=is_urdu(name),
        father_skels=name_skeletons(father) if father else [],
        addr=normalize_address(address) if address else None,
        cnic=normalize_cnic(str(query.get("cnic", "") or "")),
        phone=normalize_phone(str(query.get("phone", "") or "")),
        dob=(str(query.get("dob", "")).strip() or None),
        city=(str(query.get("city", "")).strip().lower() or None),
        raw=query,
    )


def match_record(query: dict, index: dict | None = None,
                 data_dir: str = DATA_DIR, top_k: int = 8) -> list[dict]:
    """Return up to top_k candidate entities, best posterior first."""
    from matching.blocking import block_keys

    if index is None:
        index = load_index(data_dir)
    q = _build_query(query)

    cand: set[int] = set()
    for key in block_keys(q):
        cand.update(index["block_index"].get(key, []))

    best: dict[int, dict] = {}
    for i in cand:
        r = index["records"][i]
        score, bands = index["model"].score(q, r)
        prob = index["model"].posterior(score)
        eid = index["rec2int"].get(str(r.record_id))
        if eid is None:
            continue
        if eid not in best or prob > best[eid]["posterior"]:
            best[eid] = {
                "entity_id": eid,
                "posterior": round(float(prob), 4),
                "score_bits": round(float(score), 2),
                "evidence": bands,
                "candidate_name": r.raw_name,
                "candidate_record": r.record_id,
                "registry": r.registry,
            }
    return sorted(best.values(), key=lambda d: -d["posterior"])[:top_k]


if __name__ == "__main__":
    import sys
    idx = load_index()
    res = match_record({"name": "Mohd Asif Khan", "city": "Lahore"}, index=idx)
    json.dump(res, sys.stdout, indent=2, ensure_ascii=False)
