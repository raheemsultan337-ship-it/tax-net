"""
Stage 3 — Knowledge Graph construction + feature engineering.

Builds a NetworkX graph linking resolved person-entities to their assets and
locations, then derives a per-entity feature table for the detector.

Nodes:
  person:<entity_id>      a resolved individual
  vehicle:<record_id>     a vehicle (engine_cc, value)
  property:<record_id>    a property (value)
  utility:<account_id>    a utility account (monthly bill)
  address:<city|area>     a location (shared addresses reveal hidden networks)
Edges:
  person -OWNS-> vehicle / property
  person -PAYS-> utility
  person -LOCATED_AT-> address
  person -FILED-> (declared_income, tax_paid stored on the person node)

Reads ONLY data/observable/*.csv + data/resolved/mentions.csv.
Writes:
  data/resolved/graph.gpickle      the knowledge graph
  data/resolved/entity_features.csv per-entity footprint + graph features
"""

import os
import pickle
from collections import defaultdict

import pandas as pd
import networkx as nx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBS_DIR = os.path.join(ROOT, "data", "observable")
RES_DIR = os.path.join(ROOT, "data", "resolved")


def _load():
    mentions = pd.read_csv(os.path.join(RES_DIR, "mentions.csv"))
    obs = {f: pd.read_csv(os.path.join(OBS_DIR, f"{f}.csv"))
           for f in ["vehicles", "real_estate", "utilities", "travel", "tax_returns"]}
    # record_id -> entity_id  (account_id/return_id were stored under record_id col)
    rec2ent = dict(zip(mentions.record_id, mentions.entity_id))
    return mentions, obs, rec2ent


def build():
    mentions, obs, rec2ent = _load()
    G = nx.Graph()

    feats = defaultdict(lambda: defaultdict(float))
    entity_city = defaultdict(set)

    def ent(rid):
        return rec2ent.get(rid)

    # persons
    for eid in mentions.entity_id.unique():
        G.add_node(f"person:{eid}", kind="person", entity_id=int(eid))

    # vehicles
    for r in obs["vehicles"].itertuples():
        e = ent(r.record_id)
        n = f"vehicle:{r.record_id}"
        G.add_node(n, kind="vehicle", engine_cc=int(r.engine_cc), value=int(r.reg_value))
        G.add_edge(f"person:{e}", n, rel="OWNS")
        feats[e]["n_vehicles"] += 1
        feats[e]["total_vehicle_cc"] += r.engine_cc
        feats[e]["max_vehicle_cc"] = max(feats[e]["max_vehicle_cc"], r.engine_cc)
        feats[e]["total_vehicle_value"] += r.reg_value
        entity_city[e].add(str(r.city).lower())

    # real estate
    for r in obs["real_estate"].itertuples():
        e = ent(r.record_id)
        n = f"property:{r.record_id}"
        G.add_node(n, kind="property", value=int(r.property_value))
        G.add_edge(f"person:{e}", n, rel="OWNS")
        addr = f"address:{str(r.household).lower()}"   # fine-grained household
        G.add_node(addr, kind="address")
        G.add_edge(f"person:{e}", addr, rel="LOCATED_AT")
        feats[e]["n_properties"] += 1
        feats[e]["total_property_value"] += r.property_value
        entity_city[e].add(str(r.city).lower())

    # utilities
    for r in obs["utilities"].itertuples():
        e = ent(r.account_id)
        n = f"utility:{r.account_id}"
        G.add_node(n, kind="utility", bill=int(r.avg_monthly_bill))
        G.add_edge(f"person:{e}", n, rel="PAYS")
        addr = f"address:{str(r.address).lower()}"
        G.add_node(addr, kind="address")
        G.add_edge(f"person:{e}", addr, rel="LOCATED_AT")
        feats[e]["max_monthly_bill"] = max(feats[e]["max_monthly_bill"], r.avg_monthly_bill)
        feats[e]["total_monthly_bill"] += r.avg_monthly_bill

    # travel
    for r in obs["travel"].itertuples():
        e = ent(r.record_id)
        feats[e]["intl_trips"] += int(r.trips_last_year)

    # tax returns -> stored on person node
    filed = set()
    for r in obs["tax_returns"].itertuples():
        e = ent(r.return_id)
        G.nodes[f"person:{e}"]["declared_income"] = int(r.declared_income)
        G.nodes[f"person:{e}"]["tax_paid"] = int(r.tax_paid)
        feats[e]["declared_income"] = r.declared_income
        feats[e]["tax_paid"] = r.tax_paid
        feats[e]["is_filer"] = 1.0
        filed.add(e)

    # --- graph-structural features -------------------------------------------
    # shared-address cluster: how many OTHER person-entities share an address
    # with this one (a signal of asset-hiding via family / proxy networks).
    addr_to_persons = defaultdict(set)
    for node, d in G.nodes(data=True):
        if d.get("kind") == "address":
            for nb in G.neighbors(node):
                if nb.startswith("person:"):
                    addr_to_persons[node].add(nb)

    person_cocluster = defaultdict(set)
    for addr, pers in addr_to_persons.items():
        for p in pers:
            person_cocluster[p] |= (pers - {p})

    for eid in mentions.entity_id.unique():
        f = feats[eid]
        f["n_cities"] = len(entity_city.get(eid, set())) or 1
        f["asset_node_degree"] = G.degree(f"person:{eid}")
        f["shared_address_neighbors"] = len(person_cocluster.get(f"person:{eid}", set()))
        if eid not in filed:
            f["is_filer"] = 0.0
            f["declared_income"] = 0.0
            f["tax_paid"] = 0.0

    # --- NETWORK features: attribute connected hidden wealth ------------------
    # For each person, aggregate the assets held by household-neighbours. Assets
    # held by neighbours who DON'T file (or declare almost nothing) are the
    # tell-tale of proxy/benami holdings — this is what lets the graph flag a
    # principal whose OWN books look clean.
    LOW_DECLARED = 600_000
    for eid in mentions.entity_id.unique():
        nbrs = person_cocluster.get(f"person:{eid}", set())
        net_assets = nonfiler_assets = 0.0
        for nb in nbrs:
            j = int(nb.split(":")[1])
            jf = feats[j]
            j_assets = jf.get("total_property_value", 0.0) + jf.get("total_vehicle_value", 0.0)
            net_assets += j_assets
            if jf.get("is_filer", 0.0) == 0.0 or jf.get("declared_income", 0.0) < LOW_DECLARED:
                nonfiler_assets += j_assets
        feats[eid]["network_neighbor_count"] = len(nbrs)
        feats[eid]["network_asset_value"] = net_assets
        feats[eid]["network_nonfiler_asset_value"] = nonfiler_assets

    # assemble feature frame
    cols = ["n_vehicles", "total_vehicle_cc", "max_vehicle_cc", "total_vehicle_value",
            "n_properties", "total_property_value", "max_monthly_bill",
            "total_monthly_bill", "intl_trips", "declared_income", "tax_paid",
            "is_filer", "n_cities", "asset_node_degree", "shared_address_neighbors",
            "network_neighbor_count", "network_asset_value",
            "network_nonfiler_asset_value"]
    rows = []
    for eid in mentions.entity_id.unique():
        row = {"entity_id": int(eid)}
        for c in cols:
            row[c] = feats[eid].get(c, 0.0)
        rows.append(row)
    fdf = pd.DataFrame(rows).fillna(0.0)

    with open(os.path.join(RES_DIR, "graph.gpickle"), "wb") as fh:
        pickle.dump(G, fh)
    fdf.to_csv(os.path.join(RES_DIR, "entity_features.csv"), index=False)

    print(f"  graph nodes: {G.number_of_nodes()}  edges: {G.number_of_edges()}")
    kinds = defaultdict(int)
    for _, d in G.nodes(data=True):
        kinds[d.get("kind", "?")] += 1
    print("  node kinds:", dict(kinds))
    print(f"  entity feature table: {fdf.shape[0]} entities x {fdf.shape[1]-1} features")
    print(f"  non-filers among entities: {int((fdf.is_filer==0).sum())}")
    print("  wrote data/resolved/graph.gpickle + entity_features.csv")
    return G, fdf


if __name__ == "__main__":
    build()
