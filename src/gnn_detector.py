"""
Stage 4b (optional) — GNN anomaly detector.

A graph autoencoder (GraphSAGE encoder + inner-product / MLP decoder) trained
UNSUPERVISED on the knowledge graph. Per-node reconstruction error is the
anomaly signal. This addresses the brief's "GNN for anomaly detection" academic
criterion. It is a FALLBACK / comparison layer — Isolation Forest (scoring.py)
is the production-grade core.

Why a fallback: PyTorch / PyTorch-Geometric may not ship wheels for the very
latest Python (e.g. 3.14). This module guards its imports and prints clear
guidance instead of crashing. To run it, use a Python with torch support
(<=3.13) and:  pip install torch torch-geometric

Output (when runnable): data/resolved/gnn_scores.csv  (entity_id, gnn_score)
and a printed AP comparison vs Isolation Forest.
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "data", "resolved")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv
    TORCH_OK = True
except Exception as e:                                    # noqa: BLE001
    TORCH_OK = False
    _IMPORT_ERR = e


# Same engineered mismatch signals the Isolation Forest uses, plus structural
# graph features — so the GNN comparison is apples-to-apples.
FEATURE_COLS = [
    "implied_to_declared", "elec_to_declared", "property_to_declared",
    "vehicle_to_declared", "travel_to_declared", "is_filer",
    "asset_node_degree", "n_cities",
]


def _load_pyg():
    """Build a node-level PyG graph: person entities with engineered features."""
    import scoring
    feats = scoring.engineer(pd.read_csv(os.path.join(RES, "entity_features.csv")))
    with open(os.path.join(RES, "graph.gpickle"), "rb") as fh:
        G = pickle.load(fh)

    eids = feats["entity_id"].tolist()
    idx_of = {e: i for i, e in enumerate(eids)}

    X = feats[FEATURE_COLS].values.astype(np.float32)
    # log-scale heavy-tailed columns, then standardize
    for j, c in enumerate(FEATURE_COLS):
        if c not in ("is_filer", "n_cities"):
            X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)

    # edges: connect two person-entities if they share an asset/address node
    edges = []
    for node in G.nodes():
        if isinstance(node, str) and node.startswith("person:"):
            continue
        persons = [nb for nb in G.neighbors(node) if str(nb).startswith("person:")]
        pe = [idx_of.get(int(p.split(":")[1])) for p in persons]
        pe = [p for p in pe if p is not None]
        for a in range(len(pe)):
            for b in range(a + 1, len(pe)):
                edges.append((pe[a], pe[b]))
                edges.append((pe[b], pe[a]))
    # self-loops keep isolated person nodes trainable
    for i in range(len(eids)):
        edges.append((i, i))

    edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
    data = Data(x=torch.tensor(X), edge_index=edge_index)
    return data, eids


if TORCH_OK:
    class GraphAutoEncoder(nn.Module):
        def __init__(self, in_dim, hid=32, z=16):
            super().__init__()
            self.enc1 = SAGEConv(in_dim, hid)
            self.enc2 = SAGEConv(hid, z)
            self.dec = nn.Sequential(nn.Linear(z, hid), nn.ReLU(), nn.Linear(hid, in_dim))

        def forward(self, x, ei):
            h = F.relu(self.enc1(x, ei))
            zz = self.enc2(h, ei)
            return self.dec(zz)


def run(epochs=200):
    if not TORCH_OK:
        print("GNN fallback unavailable in this environment:")
        print(f"  import error: {_IMPORT_ERR}")
        print("  Install in a torch-compatible Python (<=3.13):")
        print("    pip install torch torch-geometric")
        return None

    torch.manual_seed(42)
    data, eids = _load_pyg()
    model = GraphAutoEncoder(data.x.size(1))
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        recon = model(data.x, data.edge_index)
        loss = F.mse_loss(recon, data.x)
        loss.backward()
        opt.step()
        if (ep + 1) % 50 == 0:
            print(f"  epoch {ep+1:>3}  recon loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        recon = model(data.x, data.edge_index)
        err = ((recon - data.x) ** 2).mean(1).cpu().numpy()
    score = 100 * (err - err.min()) / (err.max() - err.min() + 1e-9)

    out = pd.DataFrame({"entity_id": eids, "gnn_score": np.round(score, 1)})
    out.to_csv(os.path.join(RES, "gnn_scores.csv"), index=False)
    print(f"\n  wrote data/resolved/gnn_scores.csv ({len(out)} entities)")
    # NB: gnn_score is reconstruction error only — no ground truth is read here.
    # All accuracy comparison (IF vs GNN vs ensemble) lives in evaluate.py, the
    # single ground-truth reader (the wall).
    return out


if __name__ == "__main__":
    run()
