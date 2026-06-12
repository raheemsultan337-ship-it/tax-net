# 🕸️ Graph AI for Broadening the National Tax Net

A knowledge-graph + entity-resolution + unsupervised-anomaly-detection pipeline
that links fragmented civic databases, identifies likely tax non-filers /
under-reporters, and produces an **explainable audit trail** for every flag.

Built for the CIKLUM hackathon problem #2 (FinTech · Knowledge Graphs · Fraud
Detection).

---

## The idea in one line

Take disconnected government databases (vehicles, real estate, utilities,
travel, tax returns), figure out which records belong to the **same person**,
build a **knowledge graph**, and flag individuals whose **lifestyle implies far
more income than they declare** — with a plain-English explanation for each.

---

## The "wall" (why the results are honest)

A strict separation prevents the system from grading its own homework:

```
data/
├── observable/        ← the DETECTOR reads ONLY this
│   ├── vehicles.csv  real_estate.csv  utilities.csv  travel.csv  tax_returns.csv
└── ground_truth/      ← read ONLY by evaluate.py, AFTER detection
    ├── persons.csv         (true income, is_evader latent flag)
    └── record_linkage.csv  (true record→person map, for ER scoring)
```

- Evasion is generated as a **latent behaviour** (independent of income and of
  how the detector scores). Its observable consequences — low declared income,
  higher chance of non-filing — are what the detector must infer from.
- The detector is **fully unsupervised**: it never sees the `is_evader` label.
  Ground truth is touched in exactly one place, `evaluate.py`, purely as a
  scorecard.

---

## Pipeline

| Stage | File | What it does |
|------|------|--------------|
| 1. Synthetic data | `src/generate_data.py` | Realistic Pakistani civic datasets with Urdu/English name variants, dirty/missing CNICs, and a latent evasion model. |
| 2. Entity resolution | `src/entity_resolution.py` | Transliteration-aware normalization, CNIC-authoritative clustering, typo recovery, city-disambiguated attachment of CNIC-less records. |
| 3. Knowledge graph | `src/build_graph.py` | NetworkX graph (persons ↔ vehicles/property/utilities/addresses) + per-entity feature table. |
| 4. Deviation scoring | `src/scoring.py` | Multi-signal **implied-income** estimator → footprint-to-declared ratios → **Isolation Forest** anomaly score (0–100) + explainable audit trail. |
| 4b. GNN fallback | `src/gnn_detector.py` | Optional graph-neural-net anomaly detector (PyTorch Geometric) for the academic-complexity criterion. |
| 4c. Score new person | `src/score_person.py` | Inference: loads the frozen model and scores a brand-new individual (deviation score + audit trail) without retraining. |
| 5. Dashboard | `app.py` | Streamlit UI. Headline KPIs (tax-base gap, proxy networks detected) + three tabs: **Flagged individuals** (ranked list, audit trails, ego-graphs), **Proxy/benami networks** (hub table, hidden-asset totals, network graph), and **Score a new individual** (enter records → live deviation score + audit trail). |
| Eval | `src/evaluate.py` | **Only** module that reads ground truth — ER + detection precision/recall. |

---

## Quick start

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows
# source .venv/bin/activate; pip install -r requirements.txt   # *nix

# run the whole pipeline (stages 1-4 + evaluation)
.venv/Scripts/python src/run_pipeline.py

# launch the dashboard
.venv/Scripts/streamlit run app.py
```

> On Windows, set `PYTHONUTF8=1` if you see encoding errors printing Urdu names.

---

## Results (default seed)

| Stage | Metric | Value |
|------|--------|-------|
Run at **N = 10,000 people** (~31,500 raw records → ~11,234 resolved entities;
full pipeline ~26 s):

| Entity resolution | Precision / Recall / F1 | **0.99 / 0.95 / 0.97** |
| | Cluster purity | **1.00** |
| Detection (Isolation Forest, own+network) | Precision@25% | **0.75** (vs ~0.31 base rate) |
| | Recall@25% | **0.60** |
| | Average Precision | **0.71** |

Entity resolution reaches near-perfect linkage by using the disambiguating fields
real civic records carry (father's name + date of birth), which separate
same-named people who share or lack a CNIC.

### The proxy/benami demonstration — why "Graph AI"

Sophisticated evaders ("principals") hide wealth by registering assets to
**proxies/frontmen** who share their household, then file a clean-looking modest
return. Their *own* books look compliant, so tabular analysis clears them. Only
the graph links them to their asset-rich, non-filing associates.

| Detector | Principal recall@25% |
|---|---|
| Isolation Forest — own features only (tabular) | **0.04** (blind) |
| Isolation Forest — own + engineered network feature | **0.51** |
| **GNN (GraphSAGE) — message-passing over the graph** | **0.93** |

**Right tool per threat.** For straightforward footprint-vs-declared mismatch
(self-evaders), Isolation Forest wins overall (AP 0.71 vs GNN 0.63) and is fully
explainable. For wealth hidden *across a network* of proxies, the **GNN dominates**
— message-passing propagates a proxy's anomalous assets back onto the principal,
something a tabular model structurally cannot do without bespoke features. The
production system uses Isolation Forest (own + network features) as the
explainable core and the GNN for relational hidden-wealth.

---

## Validation & testing

A `pytest` suite (`tests/`) runs the full pipeline once and validates:

- **Normalization & edge cases** — CNIC formatting/missing/garbage, Urdu↔English
  transliteration collapsing to one identity, empty names, typo matching.
- **Quality thresholds** — ER precision ≥ 0.95 / recall ≥ 0.90 / purity ≥ 0.97;
  detection AP ≥ 0.60; deviation scores in range; every flag has an audit trail.
- **Graph integrity** — expected node kinds and asset linkage.
- **The proxy claim** — the network signal must lift principal recall over the
  own-features baseline.
- **The wall** — observable data carries no ground-truth columns, and the
  detector modules never reference the `ground_truth/` directory.

```bash
.venv/Scripts/python -m pytest tests/ -q        # 25 tests, ~6s
```

## Scalability & deployment

Entity resolution is the only stage at risk of being O(records²). We avoid that
with **blocking on high-cardinality keys** (exact CNIC, then DOB, then name
metaphone), so only plausibly-matching records are ever compared. Measured
(`python src/benchmark.py`):

| Persons | Records | ER time | All-vs-all would be |
|--------:|--------:|--------:|--------------------:|
| 1,500 | 4,524 | 1.8 s | — |
| 3,000 | 9,149 | 3.5 s | ×4 |
| 6,000 | 18,387 | 11.6 s | ×4 |

ER grows ~linearly with data, not quadratically. For national scale (tens of
millions of records) the same blocking design ports directly to a distributed
store: persist the graph in **Neo4j / a graph DB**, run blocking + matching as a
**Spark** job, and the Isolation-Forest scoring is embarrassingly parallel per
entity. Nothing in the design assumes the data fits in memory.

## Design notes & honesty

- **Why Isolation Forest, not just a rule?** The model fits the *population*
  shape and isolates statistical outliers; legitimately wealthy **compliant**
  filers (high assets *and* high declared income) have normal footprint-to-
  declared ratios and stay inliers. We flag the *mismatch*, not wealth.
- **Explainability** is rule-derived from the model's own input features and
  population percentiles, expressed in PKR. (SHAP can be layered on where its
  dependencies are available; on Python 3.14 the percentile-based trail is the
  robust default.)
- **Known weak signal:** `shared_address_neighbors` is currently uninformative
  because synthetic addresses are coarse (city|area). Planting proxy/family
  ownership in the generator would make this a real asset-hiding signal.
