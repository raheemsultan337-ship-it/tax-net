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
| 1 | `src/generate_data.py` + `src/datagen/*` | Donor 6-registry generator adapted to tax-net's observable+GT schema; per-registry Urdu script, masked/typo CNICs, phone identifiers, address-rendering variation, real vehicle makes; DOB + father in every record; hand-seeded edge personas (demo star, father/son, twins, proxy). Latent `is_evader`/`role` labels behind the wall. SEED=42, N_PERSONS=10000. |
| 2 | `src/entity_resolution.py` + `src/matching/*` | Thin orchestrator over the ported cascade: Tier1 Fellegi-Sunter (self-calibrated, +DOB band & DOB blocking) → Tier2 multilingual sentence-embedding rescue (`models/` cache) → Tier4 graph-collective PROMOTE/SPLIT (+`shared_dob` promotion). Emits `mentions.csv` (contiguous **int** entity_id) + `match_evidence.csv`. Tier3/Ollama dropped. |
| 3 | `src/build_graph.py` | UNCHANGED. NetworkX graph + `entity_features.csv`; fine-grained household address nodes + network (proxy) features. |
| 4 | `src/scoring.py` | Multi-signal implied-income estimator → footprint/declared RATIOS + network signal → Isolation Forest (own + network). Audit trail = population-percentile reasons in PKR. |
| 4b | `src/gnn_detector.py` | GraphSAGE autoencoder. Wins on proxy networks (msg-passing). (Diagnostic GT read removed — evaluate.py is now the sole GT reader.) |
| 4c | `src/ensemble.py` | Blends IF + GNN into ONE production score (`deviation_score_combined`). Quantile-aligns GNN→IF, blends `IF_WEIGHT*IF + (1-w)*GNN`, then remaps onto IF's distribution. Falls back to IF if no torch. No ground truth (wall holds). |
| 4c | `src/score_person.py` | INFERENCE on a new individual: loads frozen `model_bundle.pkl`, predicts deviation score + audit trail without retraining. |
| 4d | `src/rule_floors.py` + `src/tax_slabs.py` | SECONDARY explainable layer (NOT the headline). Donor rule engine re-implemented over the flat `entity_features.csv`: engine-cc/electricity/property/travel income FLOORS → `entity_floors.csv` + `factors.json`. |
| 4e | `src/audit_report.py` | `AuditBuilder` → per-entity JSON + Markdown + **bilingual PDF** notice (Urdu header via Windows Arabic font + reshaper/bidi, English fallback). Headline = ensemble score; rule floors + per-link ER evidence as cross-check. `make_audits(top_n)`. |
| 5 | `app.py` + `src/live_match.py` | Streamlit dashboard: KPIs + 5 tabs — Overview (Altair), Flagged (+ floors cross-check + PDF download), Proxy/benami, **Live match** (type a record, cascade matches live with evidence), Score-a-new-individual. |
| eval | `src/evaluate.py` | ONLY reader of ground truth. Target = `is_evader OR role=='proxy'`. |
| run | `src/run_pipeline.py` | Runs stages 1–4e + eval. |
| test | `tests/` (pytest) | 42 tests: normalization (cascade primitives + DOB band), ER+detection thresholds, ensemble, graph, proxy claim, father/son + twins split, rule floors, audit, live match, the wall. `pytest tests/ -q`. |
| bench | `src/benchmark.py` | Scalability: ER near-linear via blocking. |
| pitch | `PITCH.md` | Round-2 material (value prop / market / demo script). |

## Current results (MERGED build, seed 42, N=10000 — full pipeline ~60s incl. embeddings)

- ~31,650 raw records → 10,675 resolved entities (10,000 true; ~420 proxy
  principals + ~420 proxies). Harder data than pre-merge (masked CNICs, per-registry
  Urdu, address-rendering variation) — yet every metric improved.
- **ER (cascade: Fellegi-Sunter + embeddings + graph-collective):** P **0.999** /
  R **0.968** / F1 **0.983**, cluster purity **1.000**. (Pre-merge: 0.996/0.956/0.975.)
- **Detection (Isolation Forest, own+network):** P@25% 0.77, R@25% 0.97, **AP 0.951**
  (audit-worthy base rate ~20%). (Pre-merge AP 0.79.)
- **Proxy/benami scenario (the graph payoff):** principal recall@25% —
  IF own-only **0.085**, IF own+network **0.92**, GNN (msg-passing) ~0.94.
  IF for tabular mismatch, GNN for proxy networks. Right tool per threat.
- **ENSEMBLE (one production score, `deviation_score_combined`):** AP **0.961**,
  principal recall@25% **0.954**. (Pre-merge 0.78 / 0.90.) `IF_WEIGHT=0.7`.
  Dashboard headline = combined; audit panel shows the per-model breakdown.
- **Rule-based floors (secondary, explainable):** 2,074 high-band; Rs ~13.0B
  lifestyle-implied tax gap. Used in the audit notice + dashboard cross-check.
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

## WHERE WE LEFT OFF (2026-06-13 — THE MERGE)

Merged the donor `D:\Projects\hackathon-tax-net` INTO this repo to get best-of-both.
Decisions (fixed by the user): base = tax-net; data = donor's richer 6-registry
generator enriched with DOB+father; scoring = IF+GNN ensemble stays headline, donor
rule-floors are a secondary explainable layer; the Ollama LLM tier was dropped.

What landed (all 6 workstreams, **42 pytest tests green**, dashboard HTTP 200,
full pipeline runs end-to-end ~60s):
- **Data:** `src/datagen/*` (ported names/addresses/noise/tax/personas/registries) +
  thin `src/generate_data.py`. Donor population model mapped to tax-net's GT schema
  (`is_evader` = materially under-reporting a materially-taxable income, ≥2.5M; proxy
  owner→`principal`, frontman→`proxy`). Proxy `_proxy_pair` reworked so the principal
  keeps a SMALL visible footprint incl. a utility at the **stable canonical** household
  string (so build_graph forms the shared-address link) while the non-filing proxy
  holds the bulk. build_graph UNCHANGED (column-name contract preserved).
- **ER cascade:** `src/matching/*` (normalize/features/blocking/fellegi/cluster/
  pipeline/tier2/tier4 + records). Added a **DOB evidence band**, **DOB blocking key**
  (`D:{dob}|{last_skel}`), a tier1 **dob_veto**, and a **tier4 `shared_dob` PROMOTE** —
  the last lifted recall 0.934→0.968 at precision 0.999. `entity_resolution.py` is a
  thin orchestrator emitting `mentions.csv` (string entity_id remapped to contiguous
  **int** for build_graph) + `match_evidence.csv`. Model cache copied to `models/`.
- **Rule floors / audit / dashboard / wall** — see the stage table above.
- **Wall made airtight:** removed gnn_detector's diagnostic GT read; `evaluate.py` is
  now the SOLE ground-truth reader. Wall test extended to `matching/*` + new modules.

Gotchas worth remembering: `entity_id` MUST stay int (build_graph does `int(eid)`);
the proxy graph link only forms if principal & proxy emit byte-identical household
strings (we write `Address.canonical()`, not `render_address`); tier2 needs
`sentence-transformers` + the `models/` cache (guarded — degrades to tier1+tier4 if
absent); PDF Urdu header uses Windows tahoma/arial Arabic glyphs + reshaper/bidi.
New deps installed: sentence-transformers, reportlab, arabic_reshaper, python-bidi.

---

### Earlier session (2026-06-12, part 2)

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
