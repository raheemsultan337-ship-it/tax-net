"""Tier 4: collective entity resolution over the identifier graph.

Two passes, both fully explainable:

  PROMOTE - a borderline pair gets graph context: if the two entities'
  OTHER records share hard identifiers (CNIC, phone, house address), that
  collective evidence adds bits to the pair's score. Two records that
  individually look ambiguous merge when their clusters are clearly the
  same household identity web AND the names agree.

  SPLIT - union-find transitivity can chain records of two people into one
  entity through a weak middle link. Any entity containing two or more
  distinct full CNICs is partitioned back around its CNIC cores; records
  without a CNIC follow their strongest direct edge. The graph audits its
  own clusters - no ground truth involved.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .features import band_name, compare, damerau1
from .fellegi import FellegiSunter
from .records import Record, load_records

PROMOTE_BITS = {"shared_cnic": 8.0, "shared_phone": 6.0, "shared_addr": 3.0,
                "shared_dob": 4.0}
PROMOTE_PROB = 0.97


def _addr_key(r: Record) -> str | None:
    if r.addr and r.addr["house"] is not None and r.addr["city"]:
        return f"{r.addr['city']}:{r.addr['house']}:{r.addr['street']}"
    return None


def _entity_identifiers(records: list[Record], members: dict[str, list[int]]):
    """Per entity: the sets of hard identifiers its records carry."""
    ids: dict[str, dict[str, set]] = {}
    for eid, idxs in members.items():
        cnics, phones, addrs, dobs = set(), set(), set(), set()
        for i in idxs:
            r = records[i]
            if r.cnic["digits"]:
                cnics.add(r.cnic["digits"])
            if r.phone:
                phones.add(r.phone)
            key = _addr_key(r)
            if key:
                addrs.add(key)
            if r.dob:
                dobs.add(r.dob)
        ids[eid] = {"cnic": cnics, "phone": phones, "addr": addrs, "dob": dobs}
    return ids


def run(data_dir: str = "data") -> dict:
    data = Path(data_dir)
    out_dir = data / "er_output"
    records = load_records(data_dir)
    idx_of = {r.record_id: i for i, r in enumerate(records)}

    clusters = pd.read_csv(out_dir / "clusters.csv", dtype=str)
    entity_of: dict[str, str] = dict(zip(clusters.record_id, clusters.entity_id))
    members: dict[str, list[int]] = defaultdict(list)
    for rid, eid in entity_of.items():
        members[eid].append(idx_of[rid])

    model = FellegiSunter()
    meta = json.loads((out_dir / "er_meta.json").read_text(encoding="utf-8"))
    model.weights = meta["fellegi_sunter"]["weights"]
    model.prior_logodds = meta["fellegi_sunter"]["prior_logodds"]

    borderline = pd.read_csv(out_dir / "borderline_pairs.csv", dtype=str)

    # ---- PROMOTE pass ----
    ident = _entity_identifiers(records, members)
    promotions: list[dict] = []
    merges: list[tuple[str, str]] = []
    rows_iter = [] if borderline.empty else borderline.to_dict("records")
    for row in rows_iter:
        ra, rb = records[idx_of[row["record_a"]]], records[idx_of[row["record_b"]]]
        ea, eb = entity_of[ra.record_id], entity_of[rb.record_id]
        if ea == eb:
            continue
        bands = compare(ra, rb)
        if "cnic_diff" in bands or "dob_diff" in bands:
            continue
        if band_name(ra, rb) not in ("name_exact", "name_close"):
            continue
        ia, ib = ident[ea], ident[eb]
        graph_evidence = []
        bonus = 0.0
        if ia["cnic"] & ib["cnic"]:
            bonus += PROMOTE_BITS["shared_cnic"]
            graph_evidence.append("shared_cnic")
        if ia["phone"] & ib["phone"]:
            bonus += PROMOTE_BITS["shared_phone"]
            graph_evidence.append("shared_phone")
        if ia["addr"] & ib["addr"]:
            bonus += PROMOTE_BITS["shared_addr"]
            graph_evidence.append("shared_addr")
        # Shared exact DOB across two same-name fragments is near-unique — the
        # collective recall lever for masked/missing-CNIC records.
        if ia["dob"] & ib["dob"]:
            bonus += PROMOTE_BITS["shared_dob"]
            graph_evidence.append("shared_dob")
        if not graph_evidence:
            continue
        score = sum(model.weights.get(b, 0.0) for b in bands) + bonus
        prob = model.posterior(score)
        if prob >= PROMOTE_PROB:
            merges.append((ea, eb))
            promotions.append({
                "record_a": ra.record_id, "record_b": rb.record_id,
                "entity_a": ea, "entity_b": eb,
                "evidence": "|".join(bands),
                "graph_evidence": "|".join(graph_evidence),
                "posterior_prob": round(prob, 4),
                "decided_by": "tier4_promote",
            })

    # apply merges with union-find over entity ids
    parent: dict[str, str] = {}

    def find(e: str) -> str:
        parent.setdefault(e, e)
        while parent[e] != e:
            parent[e] = parent[parent[e]]
            e = parent[e]
        return e

    for ea, eb in merges:
        ra, rb = find(ea), find(eb)
        if ra != rb:
            parent[rb] = ra
    entity_of = {rid: find(eid) for rid, eid in entity_of.items()}

    # ---- SPLIT pass ----
    members = defaultdict(list)
    for rid, eid in entity_of.items():
        members[eid].append(idx_of[rid])
    splits: list[dict] = []
    for eid, idxs in list(members.items()):
        cnics = {records[i].cnic["digits"] for i in idxs if records[i].cnic["digits"]}
        # CNICs one typo apart are the same document, not two people.
        canon: dict[str, str] = {}
        for c in sorted(cnics):
            for rep in canon.values():
                if damerau1(c, rep):
                    canon[c] = rep
                    break
            else:
                canon[c] = c
        if len(set(canon.values())) < 2:
            continue
        cores: dict[str, list[int]] = defaultdict(list)
        loose: list[int] = []
        for i in idxs:
            d = records[i].cnic["digits"]
            if d:
                cores[canon[d]].append(i)
            else:
                loose.append(i)
        for i in loose:
            best_c, best_s = None, -1e9
            for c, core in cores.items():
                s = max(sum(model.weights.get(b, 0.0)
                            for b in compare(records[i], records[j]))
                        for j in core)
                if s > best_s:
                    best_c, best_s = c, s
            cores[best_c].append(i)
        for k, (c, group) in enumerate(sorted(cores.items())):
            new_eid = eid if k == 0 else f"{eid}.S{k}"
            for i in group:
                entity_of[records[i].record_id] = new_eid
        splits.append({"entity_id": eid, "n_cnics": len(cnics),
                       "n_records": len(idxs), "n_new_entities": len(cores)})

    # ---- FATHER-CONFLICT SPLIT pass ----
    # Entities whose records disagree on father name contain two people
    # (typically father and son at the same address) chained together by an
    # identifier-poor record such as an electricity bill.
    from .features import _token_set_sim

    members = defaultdict(list)
    for rid, eid in entity_of.items():
        members[eid].append(idx_of[rid])
    for eid, idxs in list(members.items()):
        with_father = [i for i in idxs if records[i].father_skels]
        if len(with_father) < 2:
            continue
        cores: list[list[int]] = []
        for i in with_father:
            for core in cores:
                if _token_set_sim(records[i].father_skels,
                                  records[core[0]].father_skels) >= 0.75:
                    core.append(i)
                    break
            else:
                cores.append([i])
        if len(cores) < 2:
            continue
        core_cnics = [{records[i].cnic["digits"] for i in core
                       if records[i].cnic["digits"]} for core in cores]
        shared_cnic = any(core_cnics[a] & core_cnics[b]
                          for a in range(len(cores))
                          for b in range(a + 1, len(cores)))
        if shared_cnic:
            continue
        loose = [i for i in idxs if not records[i].father_skels]
        for i in loose:
            best_k, best_s = 0, -1e9
            for k, core in enumerate(cores):
                s = max(sum(model.weights.get(b, 0.0)
                            for b in compare(records[i], records[j]))
                        for j in core)
                if s > best_s:
                    best_k, best_s = k, s
            cores[best_k].append(i)
        for k, core in enumerate(cores):
            new_eid = eid if k == 0 else f"{eid}.F{k}"
            for i in core:
                entity_of[records[i].record_id] = new_eid
        splits.append({"entity_id": eid, "n_cnics": 0,
                       "n_records": len(idxs), "n_new_entities": len(cores),
                       "basis": "father_conflict"})

    # rewrite clusters
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
    pd.DataFrame(promotions).to_csv(out_dir / "tier4_promotions.csv",
                                    index=False, encoding="utf-8-sig")

    stats = {
        "promoted_pairs": len(promotions),
        "entity_merges": len(merges),
        "entities_split": len(splits),
        "n_entities_after": len(members),
    }
    meta["tier4"] = stats
    (out_dir / "er_meta.json").write_text(json.dumps(meta, indent=2),
                                          encoding="utf-8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    print(json.dumps(run(args.data_dir), indent=2))


if __name__ == "__main__":
    main()
