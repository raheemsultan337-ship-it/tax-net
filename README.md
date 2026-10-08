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

| Stage | File(s) | What it does |
|------|------|--------------|
| 1. Synthetic data | `src/generate_data.py` + `src/datagen/*` | Realistic Pakistani civic datasets: per-registry **Urdu script**, masked/typo'd CNICs, phone identifiers, address-rendering variation, real vehicle makes, DOB + father in every record; a latent evasion + proxy model behind the wall. |
| 2. Entity resolution | `src/entity_resolution.py` + `src/matching/*` | A multi-tier cascade: **Tier 1** unsupervised Fellegi-Sunter probabilistic matching (consonant-skeleton script bridging + a DOB band) → **Tier 2** multilingual sentence-embedding rescue (Urdu↔Roman) → **Tier 4** collective resolution over the identifier graph (merge/split; Tier 3 LLM arbitration was omitted for speed and offline determinism). Emits per-link evidence for the audit trail. |
| 3. Knowledge graph | `src/build_graph.py` | NetworkX graph (persons ↔ vehicles/property/utilities/addresses) + per-entity feature table. |
| 4. Deviation scoring | `src/scoring.py` | Multi-signal **implied-income** estimator → footprint-to-declared ratios → **Isolation Forest** anomaly score (0–100) + explainable audit trail. |
| 4b. GNN detector | `src/gnn_detector.py` | Graph-neural-net (GraphSAGE) anomaly detector — message-passing catches wealth hidden across proxy networks. |
| 4c. Ensemble | `src/ensemble.py` | Blends IF + GNN into one production score (`deviation_score_combined`) — the dashboard headline. Falls back to IF if torch is absent. |
| 4d. Rule floors | `src/rule_floors.py`, `src/tax_slabs.py` | Secondary, fully explainable lifestyle-income **floors** (engine-cc / electricity / property / travel) → an independent tax-gap cross-check shown in the audit + dashboard. |
| 4e. Audit notices | `src/audit_report.py` | Per-entity **bilingual (English/Urdu) PDF** notice + JSON + Markdown: headline ML score, the per-link ER evidence, and the rule-floor cross-check. |
| Inference | `src/score_person.py` | Standalone inference: scores a brand-new individual from the frozen model bundle without retraining. |
| 5. Dashboard | `app.py` | Streamlit UI: KPIs + four tabs — **Overview** (declared-vs-implied scatter with an adjustable points slider, score & tax-gap charts), **Flagged** (named entities; rich audit panel: lifestyle factors, per-link cascade evidence, observations, named records, PDF download, ego-graph), **Proxy/benami networks** (own-vs-network score lift + the named frontmen holding the assets), **Score a new individual** (live inference). |
| 6. Evaluation | `src/evaluate.py` | **Only** module that reads ground truth — ER + detection precision/recall scorecard. |

---

## Setup (after cloning)

**Prerequisites:** Python **3.12+** (developed on 3.14) and `git`. ~1 GB free disk
(a multilingual embedding model is downloaded on first run). The generated data and
the model cache are **git-ignored** — you regenerate them locally with the steps below.

### 1. Create a virtual environment & install dependencies

```bash
# from the repo root
python -m venv .venv

# Windows (PowerShell / Git Bash)
.venv/Scripts/python -m pip install --upgrade pip
.venv/Scripts/python -m pip install -r requirements.txt

# macOS / Linux
# python3 -m venv .venv && source .venv/bin/activate
# pip install --upgrade pip && pip install -r requirements.txt
```

> **PyTorch (Stage 4b GNN, optional but recommended).** If `torch` fails to install
> from PyPI on your platform, use the CPU wheels and re-run the requirements install:
> ```bash
> .venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```
> The pipeline **degrades gracefully** without torch (it falls back to the Isolation
> Forest only — the ensemble step just copies the IF score).

### 2. Run the full pipeline (generates data → resolves → scores → audits → evaluates)

```bash
# Windows — PYTHONUTF8=1 avoids cp1252 errors when printing Urdu names
set PYTHONUTF8=1                       # PowerShell: $env:PYTHONUTF8=1
.venv/Scripts/python src/run_pipeline.py

# macOS / Linux
# PYTHONUTF8=1 python src/run_pipeline.py
```

This recreates everything under `data/` (observable registries, ground truth,
resolved entities, graph, scores, rule floors) and writes bilingual audit notices to
`data/audit/`, then prints the ER + detection scorecard. At the default scale
(`N_PERSONS = 50,000` → ~195k records) a full run takes **~20 minutes** — the entity-
resolution cascade scores ~4.6M candidate pairs. To iterate faster, lower `N_PERSONS`
in `src/generate_data.py` (e.g. 10,000 ≈ ~60 s). The embedding model is cached after
the first download.

> **Entity-resolution embeddings (Tier 2).** On first run the cascade downloads
> `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (~450 MB) into
> `models/`. If the machine is **offline** and the model isn't cached, Tier 2 is
> skipped automatically (a warning prints) and resolution proceeds with Tiers 1 + 4 —
> recall drops slightly but the pipeline still completes.

### 3. Run the tests (optional but recommended)

```bash
.venv/Scripts/python -m pytest tests/ -q      # 42 tests, ~80 s
```

> Tests run at a **reduced fixed population** (8,000 people, set in `tests/conftest.py`),
> decoupled from the production `N_PERSONS`, so the suite stays fast regardless of the
> demo scale.

### 4. Launch the dashboard

```bash
.venv/Scripts/streamlit run app.py            # serves http://localhost:8501
```

The dashboard reads the artifacts produced in step 2 — **run the pipeline before
launching it** on a fresh clone.

---

## Results (default seed)

Run at **N = 50,000 people** (~195k raw records → ~53,586 resolved entities):

| Stage | Metric | Value |
|------|--------|-------|
| Entity resolution (cascade) | Precision / Recall / F1 | **1.000 / 0.972 / 0.985** |
| | Cluster purity | **1.000** |
| Detection (Isolation Forest, own+network) | Average Precision | **0.91** |
| | Precision@25% / Recall@25% | **0.90 / 0.72** (vs ~0.31 base rate) |
| **Ensemble (IF + GNN)** — production score | Average Precision | **0.90** |
| | Principal recall@25% | **0.91** |

Entity resolution reaches near-perfect linkage by combining probabilistic
Fellegi-Sunter matching, multilingual embeddings that bridge Urdu↔Roman spellings,
and collective graph resolution — using the disambiguating fields real civic records
carry (father's name + date of birth) to separate same-named people who share or lack
a CNIC.

### The proxy/benami demonstration — why "Graph AI"

Sophisticated evaders ("principals") hide wealth by registering assets to
**proxies/frontmen** who share their household, then file a clean-looking modest
return. Their *own* books look compliant, so tabular analysis clears them. Only
the graph links them to their asset-rich, non-filing associates.

| Detector | Principal recall@25% |
|---|---|
| Isolation Forest — own features only (tabular) | **0.07** (blind) |
| Isolation Forest — own + engineered network feature | **0.84** |
| **Ensemble (IF + GNN) — the production score** | **0.91** |

**Right tool per threat.** For straightforward footprint-vs-declared mismatch
(self-evaders), Isolation Forest is accurate (AP 0.91) and fully explainable. For
wealth hidden *across a network* of proxies, the **GNN dominates** — message-passing
propagates a proxy's anomalous assets back onto the principal, something a tabular
model structurally cannot do without bespoke features. The production system
**ensembles** the two: it gives up almost nothing in overall AP (0.90 vs 0.91) while
raising principal recall to 0.91, versus 0.84 for Isolation Forest with the network
feature and 0.07 without it.

---

## Validation & testing

A `pytest` suite (`tests/`) runs the full pipeline once and validates:

- **Cascade primitives & edge cases** — consonant-skeleton script bridging
  (Urdu↔Roman collapse to one key), CNIC parsing (full / masked / garbage), the DOB
  evidence band, typo (Damerau-1) matching, honorific stripping.
- **Quality thresholds** — ER precision ≥ 0.95 / recall ≥ 0.90 / purity ≥ 0.97;
  detection AP ≥ 0.60; ensemble AP ≥ 0.70; deviation scores in range; audit trails.
- **Hard ER cases** — same-name father/son and twins must stay **distinct** entities.
- **Graph integrity** — expected node kinds and asset linkage.
- **The proxy claim** — the network signal must lift principal recall over the
  own-features baseline.
- **Rule floors & audit** — the false-positive guard (compliant filers score zero
  gap) and that bilingual JSON/Markdown/PDF notices generate for the top flags.
- **The wall** — observable data carries no ground-truth columns, and every detector
  module (incl. the cascade and rule/audit layers) never references `ground_truth/`.

```bash
.venv/Scripts/python -m pytest tests/ -q        # 42 tests, ~80s
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

ER scaling is roughly linear to mildly superlinear, far below quadratic. For national scale (tens of
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
