# CLAUDE.md — Project context for `tax-net`

Context for working on this project. Read before making changes.

## What this is

Hackathon prototype (CIKLUM problem #2): **Graph AI for Broadening the National
Tax Net.** Links fragmented Pakistani civic databases, resolves identities into
a knowledge graph, and flags likely tax non-filers / under-reporters with an
**explainable audit trail** and a 0–100 Tax Compliance Deviation Score.

## Non-negotiable design principles (decided with the user)

1. **The wall.** The detector reads ONLY `data/observable/*.csv`. Ground truth
   (`data/ground_truth/`) is opened in exactly ONE place — `src/evaluate.py` —
   and only AFTER detection, as a scorecard. Never let the label influence a score.
2. **No circularity.** Evasion is a *latent behaviour* in the generator, drawn
   independently of income and of how the detector scores people. Its observable
   consequences (low declared income, higher non-filing rate) are what the ML
   must infer from. Do NOT generate evaders using the same rule the detector uses.
3. **ML must compute the result.** Detection is fully unsupervised (Isolation
   Forest). The answer key is never a feature. The user cares about this.
4. **Interpretability is a first-class goal** (it's an explicit eval criterion).

## Architecture / pipeline

| Stage | File | Notes |
|------|------|------|
| 1 | `src/generate_data.py` | Synthetic data; latent `is_evader` flag; Urdu/English name variants; dirty/missing CNICs. SEED=42, N_PERSONS=10000 (demo scale; ~4.5 records/person). |
| 2 | `src/entity_resolution.py` | CNIC-authoritative clustering (NOT transitive name edges — that over-merged). Typo recovery + attach blocked by DOB (high-cardinality → scalable) then name metaphone. DOB + father's name disambiguate CNIC-less attach. |
| 3 | `src/build_graph.py` | NetworkX graph + `entity_features.csv`; fine-grained household address nodes + network (proxy) features. |
| 4 | `src/scoring.py` | Multi-signal implied-income estimator → footprint/declared RATIOS + network signal → Isolation Forest (own + network). Audit trail = population-percentile reasons in PKR. |
| 4b | `src/gnn_detector.py` | GraphSAGE autoencoder. Wins on proxy networks (msg-passing). |
| 4c | `src/ensemble.py` | Blends IF + GNN into ONE production score (`deviation_score_combined`). Quantile-aligns GNN→IF, blends `IF_WEIGHT*IF + (1-w)*GNN`, then remaps onto IF's distribution so the 0–100 scale/threshold keep meaning. Falls back to IF if no torch. No ground truth (wall holds). |
| 4c | `src/score_person.py` | INFERENCE on a new individual: loads frozen model bundle (saved by scoring.py to `model_bundle.pkl`), predicts deviation score + audit trail without retraining. |
| 5 | `app.py` | Streamlit dashboard: KPIs + 3 tabs — Flagged, Proxy/benami networks, "Score a new individual" form (live inference). |
| eval | `src/evaluate.py` | ONLY reader of ground truth. |
| run | `src/run_pipeline.py` | Runs stages 1–4c (incl. GNN + ensemble) + eval. |
| test | `tests/` (pytest) | 26 tests: normalization/edge cases, ER+detection quality thresholds, ensemble (AP held + proxy recall lift), graph integrity, proxy claim, the wall. `pytest tests/ -q`. |
| bench | `src/benchmark.py` | Scalability: ER near-linear via blocking (restores N=1500 after). |
| pitch | `PITCH.md` | Round-2 material (value prop / market / demo script). |

## Current results (seed 42, N=10000 — full pipeline ~30s)

- ~31,500 raw records → ~11,354 resolved entities (10,440 true; 315 proxy-using
  principals).
- **ER:** P 1.00 / R 0.96 / F1 0.97, cluster purity 0.998.
- **Detection (Isolation Forest, own+network):** P@25% 0.82, R@25% 0.64, **AP 0.79**
  (audit-worthy base rate ~32%).
- **Proxy/benami scenario (the graph payoff):** principal recall@25% —
  IF own-only **0.11**, IF own+network **0.78**, GNN (msg-passing) **0.94**.
  GNN loses overall AP but DOMINATES on relational hidden-wealth. Conclusion:
  IF for tabular mismatch (self-evaders), GNN for proxy networks. Right tool per threat.
- **ENSEMBLE (one production score, `deviation_score_combined`):** AP **0.78**
  (IF 0.79 — held), principal recall@25% **0.90** (vs IF+net 0.78). Best of both:
  keeps IF's accuracy, inherits most of the GNN's proxy recall. `IF_WEIGHT=0.7`
  (tunable in ensemble.py): ↑ favours tabular/non-zero-filer catches, ↓ favours
  relational recall. Dashboard headline = combined; audit panel shows the per-model
  (tabular vs relational) breakdown + which signal drove each flag.
- **Flagged-list composition (score ≥ 60):** ~534 flagged, 21% non-zero-declared
  under-reporting filers (was 11%); 839 filers score ≥40 → visible colour in the
  declared-vs-implied scatter. Non-filers ~15% of population.
- (At N=1500 earlier: ER F1 0.98, detection AP 0.74 — metrics are stable across scale.)

### ER recall investigation (how we got from R 0.77 -> 0.98)
- Diagnosed losses: ~398 missing-CNIC records + ~119 fragmentation artifacts.
- Proved CNIC-less attach is unrecoverable with name+city alone: argmax attach
  precision was only **0.16** (same-name people are genuinely indistinguishable).
- Multi-key blocking alone didn't help (turned blocking-misses into margin-rejects).
- Fix: enriched records with **father's name + date of birth** (realistic — NADRA/FBR
  records carry both). DOB is near-unique among same-name people, so attach scoring
  (`attach_score` in entity_resolution.py) separates the true core from decoys and
  the strict margin passes at high precision. CNIC-less attached 99 -> 503.

## Key decisions & why

- **IF on RATIO features only** (`implied_to_declared`, `*_to_declared`, `is_filer`).
  Absolute magnitude features made IF flag legitimately-rich *compliant* filers.
  Ratios isolate the mismatch; compliant rich have normal ratios → inliers.
- **Income estimator** inverts the generator's footprint relationships using
  domain priors (NOT fitted on ground truth): `corr(log implied, log true)=0.96`.
  This was the single biggest detection-quality lever (AP 0.44 → 0.54).
- **ER over-merging** was the first big trap: naive transitive name-matching gave
  P 0.10 (one entity swallowed 16 people). CNIC-as-key fixed it.
- **Compliance is a SPECTRUM** (generate_data): 52% honest (ratio 0.85–1.0, ~2%
  non-file), 23% mild→moderate under-reporters that FILE (ratio 0.45–0.80), 25%
  **serious evaders — WEALTHY** (true_income ×2.0–4.5 so they have a big visible
  footprint, ratio 0.04–0.22, only ~10% non-file → they show up as flaggable
  NON-ZERO-declared filers, not just non-filers). `is_evader = declared <
  0.5*true_income` (ground-truth def; detector never sees it). Income
  lognormal(14, 1.05). Realistic premise: serious evasion concentrates among high
  earners, who have the most to hide AND the footprint that gives them away.
- **`DECL_FLOOR` raised 50k → 1.2M** (scoring.py). The floor is the imputed minimum
  income for anyone in the asset databases. At 50k, declared==0 cases got an
  unbounded footprint/declared ratio and *structurally* dominated every flag — no
  realistic under-reporting-filer population could compete (proven via a 4-config
  generator sweep). Raising it to ~population-median let under-reporting filers
  surface: non-zero share of the flagged list went 11% → 21%, AP held ~0.79,
  proxy recall unaffected. **This is rank-preserving for finite ratios but
  compresses the declared==0 advantage** — it does NOT touch the wall or the model.
- **Non-filers still rank high** (declared≈0), but the model correctly orders by
  wealth. The dashboard **segments the flagged tab** into "Wealthy non-filers" vs
  "Under-reporting filers" so both evasion types are visible.
- **Proxy/benami model** (generate_data `_inject_proxies`): `asset_income` (NOT
  true_income) drives the observable footprint. Principals shrink their visible
  footprint + file a clean matching return; proxies (same surname, shared
  `household`) hold the hidden assets and don't file. Link = fine-grained
  `household` address node. `is_evader`=True for principals, False for proxies;
  evaluate's target = `is_evader OR role=='proxy'`.
- **Network features** (build_graph): `network_nonfiler_asset_value` = assets held
  by non-filing household-neighbours. Feeds `network_to_declared` in scoring. The
  GNN gets these via message-passing natively (own features only).

## Environment / gotchas

- **Python 3.14**, venv at `D:\Projects\tax-net\.venv`. Use
  `.venv\Scripts\python.exe`. torch 2.12+cpu and torch-geometric 2.8 DO install
  on 3.14 (torch via `--index-url https://download.pytorch.org/whl/cpu`).
- **Windows console can't print Urdu** (cp1252). Prefix runs with `PYTHONUTF8=1`
  or you get UnicodeEncodeError.
- shap/numba not installed — the audit trail is percentile-based by design (no
  numba dependency risk). SHAP is optional future work.

## WHERE WE LEFT OFF (last session — 2026-06-12, part 2)

Everything is DONE, working, committed-to-disk at **N=10000**, all **25 pytest
tests green**, dashboard boots clean (HTTP 200). Data on disk is the canonical 10k
set (pytest regenerated it; deterministic seed 42).

This session (frontend + flaggable-mix work):
- **Added charts to the dashboard** (Altair, ships with Streamlit — no new dep):
  new **📈 Overview tab** (first tab) with the headline **declared-vs-implied
  log-log scatter** (evaders fall below the diagonal; colour = deviation), a
  **score-distribution histogram** (threshold marked), and a **tax-gap-by-score-band
  bar**. Plus an inline **declared-vs-implied bar** in the flagged audit panel and a
  **population-placement histogram** in the "Score a new individual" tab. Histograms
  are **pre-binned in pandas** (Altair trips its 5000-row guard at N=11k otherwise).
- **Made more non-zero-declared filers flaggable** (user request — the flagged list
  was ~90% declared==0 non-filers). Two changes: generator now models serious
  evaders as **wealthy filers** (income ×2.0–4.5, ratio 0.04–0.22, mostly FILE), and
  **`DECL_FLOOR` 50k → 1.2M** in scoring.py (the actual lever — see Key decisions).
  Non-zero share of flagged went **11% → 21%**; ~839 filers now score ≥40 (visible
  colour in the scatter). Proved data-only can't fix it (floor dominates).
- Bounded the **proxy population** back to ~315 principals (raised wealth threshold
  6M, use-prob 0.35) — the income boost had otherwise exploded it to ~1054.

Also **ensembled IF + GNN into ONE production score** (`src/ensemble.py`, stage 4c):
quantile-align GNN→IF, blend (`IF_WEIGHT=0.7`), remap onto IF's distribution so the
0–100 scale/threshold are preserved. Combined AP 0.78 (held), principal recall
0.78→**0.90**. Dashboard headline is now the combined score (`load()` aliases
`deviation_score_combined`→`deviation_score`); audit panel shows the tabular-vs-
relational breakdown. Wired into run_pipeline (4b GNN, 4c ensemble) + conftest;
added an ensemble pytest (now **26 tests**). Note: combined flagged-list non-zero
share is ~15% (vs IF-only 21%) — the GNN correctly pulls in declared=0 proxy
frontmen, which compete for slots; raise `IF_WEIGHT` to favour non-zero filers.

Current 10k headline: ER 1.00/0.96/0.97; detection **AP 0.79** (IF) / **0.78**
(combined); principal recall 0.11→0.78 (IF+net)→**0.90** (ensemble)→0.94 (GNN-only).
Re-run the dashboard to refresh the KPI rupee figures.

**Open / candidate next steps (none committed):**
- "Ingest raw records" mode: add a new person's RAW records and run them through
  the full pipeline (ER + graph + score), vs the current direct-footprint scoring.
- Richer GNN (node-type-aware / heterogeneous); add father-name graph edges as a
  second proxy link.
- Turn `PITCH.md` into actual slides.
- If still wanted: push non-zero flagged share higher (raise DECL_FLOOR further —
  sweep showed 2M → 25%; tradeoff is fewer non-filers flagged).

## Rubric alignment (CUST 2026)

Round 1 — Relevance 20% (hits every problem element), Innovation 20% (proxy/benami
graph detection), Technical 30% (full pipeline + dashboard + GNN + tests),
Validation 15% (`tests/` 25 tests + held-out P/R scorecard), Feasibility 15%
(benchmark shows near-linear ER; Neo4j+Spark scale story in README). Round 2 —
see `PITCH.md` (value prop, GovTech + bank-AML market, demo script).

When asked to improve "for the rubric": Validation = add/strengthen tests;
Feasibility = scale evidence + deployment story; Round 2 = PITCH.md.

## Conventions

- Keep the wall intact in any new code. If a new module needs ground truth, it
  belongs in / behind `evaluate.py`.
- Match existing style: stdlib + pandas/numpy/sklearn/networkx; clear docstrings
  explaining the *why*.
