"""
Stage 5 — Tax Compliance Deviation dashboard (Streamlit).

Run:  .venv/Scripts/streamlit run app.py

Reads the pipeline outputs in data/resolved/ + observable records. The hidden
ground truth is loaded ONLY to render the evaluation panel (precision/recall) —
it never influences a score.
"""

import os
import sys
import json
import pickle

import pandas as pd
import altair as alt
import streamlit as st
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from score_person import score_person   # noqa: E402  (inference on a new individual)
import live_match                         # noqa: E402  (live ER on a typed record)
import audit_report                       # noqa: E402  (bilingual PDF notice)
RES = os.path.join(ROOT, "data", "resolved")
OBS = os.path.join(ROOT, "data", "observable")
GT = os.path.join(ROOT, "data", "ground_truth")

st.set_page_config(page_title="Tax Net — Graph AI", layout="wide", page_icon="🕸️")

NODE_COLOR = {"vehicle": "#457b9d", "property": "#2a9d8f",
              "utility": "#e9c46a", "address": "#8d99ae"}


# --------------------------------------------------------------------------
@st.cache_data
def load():
    scores = pd.read_csv(os.path.join(RES, "entity_scores.csv"))
    # The production score is the ENSEMBLE (Isolation Forest + GNN). Alias it to
    # deviation_score so every KPI / tab / chart uses the combined number, while
    # keeping the per-model components for the interpretability breakdown.
    if "deviation_score_combined" in scores.columns:
        scores["deviation_score_if"] = scores["deviation_score"]
        scores["deviation_score"] = scores["deviation_score_combined"]
    mentions = pd.read_csv(os.path.join(RES, "mentions.csv"))
    feats = pd.read_csv(os.path.join(RES, "entity_features.csv")).set_index("entity_id")
    return scores, mentions, feats


@st.cache_resource
def load_graph():
    with open(os.path.join(RES, "graph.gpickle"), "rb") as fh:
        return pickle.load(fh)


@st.cache_data
def load_floors():
    """Rule-based lifestyle-floor cross-check (secondary, explainable layer)."""
    try:
        f = pd.read_csv(os.path.join(RES, "entity_floors.csv"))
        return f.set_index("entity_id")
    except FileNotFoundError:
        return None


@st.cache_resource
def load_match_index():
    return live_match.load_index()


@st.cache_resource
def audit_builder():
    return audit_report.AuditBuilder()


@st.cache_data
def eval_metrics():
    """Compute ER + detection metrics (reads ground truth — display only)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("ev", os.path.join(ROOT, "src", "evaluate.py"))
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev.evaluate_entity_resolution(), ev.evaluate_detection()


def pkr(x):
    x = float(x)
    if x >= 1e7:
        return f"Rs {x/1e7:.2f} cr"
    if x >= 1e5:
        return f"Rs {x/1e5:.2f} lakh"
    return f"Rs {x:,.0f}"


_FLOOR = 10_000          # log-axis floor so non-filers (declared 0) stay visible
_BANDS = [0, 20, 40, 60, 80, 100]
_BAND_LABELS = ["0–20", "20–40", "40–60", "60–80", "80–100"]


def _score_bins(scores, bins=40):
    """Pre-bin scores in pandas (keeps the chart payload tiny — Altair otherwise
    inlines every row and trips its 5000-row guard at N=11k)."""
    cats = pd.cut(scores.deviation_score, bins=bins)
    g = scores.groupby(cats, observed=False).size().reset_index(name="count")
    g["lo"] = g.deviation_score.map(lambda i: i.left)
    g["hi"] = g.deviation_score.map(lambda i: i.right)
    return g[["lo", "hi", "count"]]


def _hist_layer(scores, color):
    binned = _score_bins(scores)
    return alt.Chart(binned).mark_bar(color=color, opacity=0.85).encode(
        x=alt.X("lo:Q", title="Tax Compliance Deviation Score"),
        x2="hi:Q",
        y=alt.Y("count:Q", title="Entities"),
        tooltip=[alt.Tooltip("lo:Q", title="From", format=".0f"),
                 alt.Tooltip("hi:Q", title="To", format=".0f"),
                 alt.Tooltip("count:Q", title="Entities")])


def chart_score_hist(scores, threshold=60):
    """Population distribution of deviation scores, with the audit threshold marked."""
    rule = alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(
        color="#e63946", strokeDash=[6, 4], size=2).encode(x="t:Q")
    return (_hist_layer(scores, "#457b9d") + rule).properties(height=260)


def chart_declared_vs_implied(scores, sample=3500):
    """The core narrative: declared vs lifestyle-implied income (log-log). Honest
    filers sit on the diagonal; evaders fall far below it (declared << implied)."""
    df = scores.copy()
    if len(df) > sample:
        df = df.sample(sample, random_state=42)
    df["declared_plot"] = df.declared_income.clip(lower=_FLOOR)
    df["implied_plot"] = df.implied_income.clip(lower=_FLOOR)
    df["Filing"] = df.is_filer.map({1: "Filer", 0: "Non-filer"})
    lim = [_FLOOR, max(scores.implied_income.max(), scores.declared_income.max()) * 1.1]
    pts = alt.Chart(df).mark_circle(opacity=0.45).encode(
        x=alt.X("declared_plot:Q", scale=alt.Scale(type="log", domain=lim),
                title="Declared income (PKR, log)"),
        y=alt.Y("implied_plot:Q", scale=alt.Scale(type="log", domain=lim),
                title="Lifestyle-implied income (PKR, log)"),
        color=alt.Color("deviation_score:Q",
                        scale=alt.Scale(scheme="redyellowgreen", reverse=True, domain=[0, 100]),
                        title="Deviation"),
        shape=alt.Shape("Filing:N", title="Filing status"),
        tooltip=["entity_id", alt.Tooltip("deviation_score:Q", format=".0f"),
                 "declared_income", "implied_income", "Filing"],
    )
    diag = alt.Chart(pd.DataFrame({"x": lim, "y": lim})).mark_line(
        color="#444", strokeDash=[5, 5], size=1).encode(x="x:Q", y="y:Q")
    return (pts + diag).properties(height=420).interactive()


def chart_tax_gap_by_band(scores):
    """Where the recoverable tax base concentrates, bucketed by deviation score."""
    df = scores.copy()
    df["gap"] = (df.implied_income - df.declared_income).clip(lower=0)
    df["band"] = pd.cut(df.deviation_score, bins=_BANDS, labels=_BAND_LABELS,
                        include_lowest=True)
    agg = df.groupby("band", observed=False).gap.sum().reset_index()
    agg["gap_cr"] = agg.gap / 1e7
    return alt.Chart(agg).mark_bar().encode(
        x=alt.X("band:N", title="Deviation score band", sort=_BAND_LABELS),
        y=alt.Y("gap_cr:Q", title="Implied tax-base gap (Rs crore)"),
        color=alt.Color("band:N", scale=alt.Scale(scheme="reds"), legend=None),
        tooltip=[alt.Tooltip("band:N", title="Band"),
                 alt.Tooltip("gap_cr:Q", title="Gap (Rs cr)", format=".1f")],
    ).properties(height=260)


def chart_placement(scores, score):
    """Histogram of population scores with the queried individual marked."""
    rule = alt.Chart(pd.DataFrame({"s": [score]})).mark_rule(
        color="#e63946", size=3).encode(
        x="s:Q", tooltip=[alt.Tooltip("s:Q", title="This individual", format=".0f")])
    return (_hist_layer(scores, "#adb5bd").properties(height=220) + rule)


def render_network(G, feats, focal_eid, radius=3, height=460):
    """Interactive household/asset graph around one entity. Highlights co-household
    'associates' (proxies) — purple, with non-filers ringed red."""
    pnode = f"person:{focal_eid}"
    if pnode not in G:
        st.info("This entity has no graph node.")
        return
    ego = nx.ego_graph(G, pnode, radius=radius)
    net = Network(height=f"{height}px", width="100%", bgcolor="#ffffff", directed=False)
    for n, d in ego.nodes(data=True):
        k = d.get("kind", "?")
        if k == "person":
            e = d.get("entity_id")
            is_filer = int(feats.loc[e, "is_filer"]) if e in feats.index else 1
            if n == pnode:
                net.add_node(n, label=f"🎯 TARGET #{e}", color="#e63946", size=36,
                             borderWidth=3, title=f"focal entity #{e}")
            else:
                ring = "#c1121f" if is_filer == 0 else "#9d4edd"
                tag = "non-filer associate" if is_filer == 0 else "associate"
                net.add_node(n, label=f"👤 #{e}", color="#c77dff", size=24,
                             borderWidth=3, borderWidthSelected=4,
                             title=f"{tag} (entity #{e})")
        else:
            val = d.get("value") or d.get("bill") or ""
            label = k + (f"\n{pkr(val)}" if val else "")
            net.add_node(n, label=label, color=NODE_COLOR.get(k, "#999"),
                         size=16, title=f"{k} {pkr(val) if val else ''}")
    for a, b, d in ego.edges(data=True):
        net.add_edge(a, b, title=d.get("rel", ""), color="#cccccc")
    net.set_options('{"physics":{"barnesHut":{"springLength":130,"gravitationalConstant":-8000}}}')
    net.save_graph("_ego.html")
    components.html(open("_ego.html", encoding="utf-8").read(), height=height + 20)


# --------------------------------------------------------------------------
scores, mentions, feats = load()
G = load_graph()
floors = load_floors()

# derived: suspected proxy-network hubs = filers whose household associates hold
# large non-filed assets (principals file clean returns, so is_filer == 1).
scores["hidden_assets"] = scores.get("network_nonfiler_assets", 0)
scores["n_assoc"] = scores.get("network_neighbors", 0)
hubs = scores[(scores.is_filer == 1) & (scores.hidden_assets > 2_000_000)].copy()

st.title("🕸️ Graph AI — Broadening the National Tax Net")
st.caption("Entity resolution → knowledge graph → **ensemble** deviation score "
           "(Isolation Forest for tabular mismatch + GNN for hidden proxy networks), "
           "with explainable audit trails.")

# ---- headline KPIs -------------------------------------------------------
flagged = scores[scores.deviation_score >= 60]
tax_gap = (flagged.implied_income - flagged.declared_income).clip(lower=0).sum()
k = st.columns(4)
k[0].metric("Entities analysed", f"{len(scores):,}")
k[1].metric("Flagged for audit (score ≥ 60)", f"{len(flagged):,}")
k[2].metric("Estimated tax-base gap", pkr(tax_gap))
k[3].metric("Proxy networks detected", f"{len(hubs):,}",
            help="Filers whose household associates hold large undeclared assets")

with st.expander("📊 Pipeline performance (evaluated against held-out ground truth)"):
    try:
        er, det = eval_metrics()
        c = st.columns(6)
        c[0].metric("ER Precision", f"{er['precision']:.2f}")
        c[1].metric("ER Recall", f"{er['recall']:.2f}")
        c[2].metric("ER F1", f"{er['f1']:.2f}")
        if det:
            c[3].metric("Detect P@25%", f"{det['precision']:.2f}")
            c[4].metric("Detect R@25%", f"{det['recall']:.2f}")
            c[5].metric("Avg Precision", f"{det['ap']:.2f}")
        st.caption("Ground truth is read ONLY here, after detection — it never feeds the detector.")
    except Exception as e:
        st.warning(f"Could not compute metrics: {e}")

tab_overview, tab_flagged, tab_proxy, tab_live, tab_new = st.tabs(
    ["📈 Overview", "🚩 Flagged individuals", "🕵️ Proxy / benami networks",
     "🔎 Live match", "➕ Score a new individual"])

# ==========================================================================
with tab_overview:
    st.subheader("Declared vs. lifestyle-implied income")
    st.markdown(
        "Each point is a resolved individual. Honest filers sit on the **dashed "
        "diagonal** (declared ≈ implied). The further an entity falls **below** the "
        "line, the larger the gap between what they spend and what they declare — "
        "redder points are the model's higher-deviation flags.")
    st.altair_chart(chart_declared_vs_implied(scores), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Score distribution")
        st.caption("Where the population lands. Red dashed line = audit threshold (60).")
        st.altair_chart(chart_score_hist(scores), use_container_width=True)
    with c2:
        st.subheader("Where the tax gap concentrates")
        st.caption("Summed implied-vs-declared gap, bucketed by deviation score.")
        st.altair_chart(chart_tax_gap_by_band(scores), use_container_width=True)

# ==========================================================================
with tab_flagged:
    # Two distinct evasion types, surfaced separately so non-filers don't drown
    # out the subtler under-reporting filers.
    nonfilers = scores[scores.declared_income == 0]
    underfilers = scores[scores.declared_income > 0]
    seg = st.radio(
        "Evasion type",
        [f"🚫 Wealthy non-filers ({len(nonfilers[nonfilers.deviation_score>=60])})",
         f"⚠️ Under-reporting filers ({len(underfilers[underfilers.deviation_score>=45])})",
         "All flagged"],
        horizontal=True,
        help="Non-filers declared nothing; under-reporting filers declared some "
             "income but their lifestyle implies far more.")

    left, right = st.columns([1, 1.3])
    with left:
        if seg.startswith("🚫"):
            pool, default_min = nonfilers, 60
            st.caption("People who filed **no return** but own significant assets.")
        elif seg.startswith("⚠️"):
            pool, default_min = underfilers, 40
            st.caption("People who **filed** but declared far less than their lifestyle implies.")
        else:
            pool, default_min = scores, 60
        min_score = st.slider("Minimum deviation score", 0, 100, default_min, key=seg)
        filt = pool[pool.deviation_score >= min_score].copy()
        filt["declared"] = filt.declared_income.map(pkr)
        filt["implied"] = filt.implied_income.map(pkr)
        filt["filer"] = filt.is_filer.map({1: "filer", 0: "NON-FILER"})
        st.dataframe(
            filt[["entity_id", "deviation_score", "declared", "implied", "filer"]]
            .rename(columns={"entity_id": "Entity", "deviation_score": "Score"}),
            height=430, use_container_width=True, hide_index=True,
        )
        st.caption(f"{len(filt)} entities at/above this score in this category.")

    with right:
        st.subheader("🔍 Audit trail")
        options = filt.sort_values("deviation_score", ascending=False).entity_id.tolist()
        if not options:
            st.info("No entities at this score threshold.")
        else:
            eid = st.selectbox("Select a flagged entity", options,
                               format_func=lambda e: f"Entity #{e}  (score "
                               f"{scores.loc[scores.entity_id==e,'deviation_score'].iloc[0]:.0f})")
            row = scores[scores.entity_id == eid].iloc[0]
            m = st.columns(3)
            m[0].metric("Deviation Score", f"{row.deviation_score:.0f}/100",
                        help="Production score: ensemble of the tabular (Isolation "
                             "Forest) and relational (GNN) detectors.")
            m[1].metric("Declared income", pkr(row.declared_income))
            m[2].metric("Lifestyle-implied", f"~{pkr(row.implied_income)}")
            if "deviation_score_if" in row and pd.notna(row.get("gnn_score")):
                b = st.columns(2)
                b[0].metric("🧾 Tabular signal (IF)", f"{row.deviation_score_if:.0f}",
                            help="Own footprint vs declared income mismatch.")
                b[1].metric("🕸️ Relational signal (GNN)", f"{row.gnn_score:.0f}",
                            help="Anomaly inferred from household/asset network "
                                 "(catches wealth hidden behind proxies).")
                driver = ("the relational/network signal" if row.gnn_score > row.deviation_score_if
                          else "the tabular footprint-vs-declared signal")
                st.caption(f"Primary driver of this flag: **{driver}**.")
            gap_df = pd.DataFrame({
                "kind": ["Declared", "Lifestyle-implied"],
                "value": [row.declared_income, row.implied_income]})
            st.altair_chart(
                alt.Chart(gap_df).mark_bar().encode(
                    x=alt.X("value:Q", title="PKR"),
                    y=alt.Y("kind:N", title="", sort=["Declared", "Lifestyle-implied"]),
                    color=alt.Color("kind:N", scale=alt.Scale(
                        domain=["Declared", "Lifestyle-implied"],
                        range=["#457b9d", "#e63946"]), legend=None),
                    tooltip=["kind", "value"]).properties(height=110),
                use_container_width=True)
            st.markdown("**Why this entity was flagged (ML signal):**")
            for reason in json.loads(row.audit_trail):
                st.markdown(f"- {reason}")

            # Secondary, deterministic cross-check: lifestyle-implied income floors.
            if floors is not None and eid in floors.index:
                fr = floors.loc[eid]
                with st.expander("🧮 Explainable cross-check — lifestyle income floors "
                                 "(secondary, not the headline)"):
                    f = st.columns(3)
                    f[0].metric("Estimated income (floor)", pkr(fr.estimated_income_pkr))
                    f[1].metric("Expected tax", pkr(fr.expected_tax_pkr))
                    f[2].metric("Lifestyle-implied tax gap", pkr(fr.tax_gap_pkr),
                                help=f"rule band: {fr.band}")
                    st.caption("Deterministic rule engine (engine-cc / electricity / "
                               "property / travel floors) — an independent sanity check "
                               "on the ML ranking, and the basis of the audit notice.")

            # Downloadable bilingual audit notice (PDF).
            try:
                builder = audit_builder()
                audit = builder.build(eid)
                pdf_path = os.path.join(ROOT, "data", "audit", f"_dash_{eid}.pdf")
                builder.pdf(audit, pdf_path)
                with open(pdf_path, "rb") as fh:
                    st.download_button("📄 Download audit notice (PDF)", fh.read(),
                                       file_name=f"tax_notice_{eid}.pdf",
                                       mime="application/pdf")
            except Exception as e:
                st.caption(f"(PDF notice unavailable: {e})")

            recs = mentions[mentions.entity_id == eid]
            with st.expander(f"📄 {len(recs)} source records across "
                             f"{recs.source.nunique()} databases"):
                st.dataframe(recs[["source", "record_id", "raw_name", "cnic", "city"]],
                             hide_index=True, use_container_width=True)

    if options:
        st.subheader(f"🌐 Knowledge graph — Entity #{eid}")
        render_network(G, feats, eid, radius=1)

# ==========================================================================
with tab_proxy:
    st.subheader("🕵️ Suspected proxy / benami networks")
    st.markdown(
        "These individuals file **clean-looking returns**, but the graph links them "
        "to household associates holding large **undeclared** assets — wealth hidden "
        "behind frontmen. *Tabular analysis alone misses these; the network reveals them.*"
    )
    if not len(hubs):
        st.info("No proxy networks detected at the current data/threshold.")
    else:
        h = hubs.sort_values("hidden_assets", ascending=False).copy()
        h["declared_"] = h.declared_income.map(pkr)
        h["hidden_"] = h.hidden_assets.map(pkr)
        st.dataframe(
            h[["entity_id", "deviation_score", "declared_", "n_assoc", "hidden_"]]
            .rename(columns={"entity_id": "Principal", "deviation_score": "Score",
                             "declared_": "Declared", "n_assoc": "Associates",
                             "hidden_": "Hidden assets (via associates)"}),
            height=240, use_container_width=True, hide_index=True,
        )
        st.metric("Total hidden assets uncovered via networks",
                  pkr(h.hidden_assets.sum()))

        pid = st.selectbox("Inspect a network", h.entity_id.tolist(),
                           format_func=lambda e: f"Principal #{e}  "
                           f"(hidden ~{pkr(scores.loc[scores.entity_id==e,'hidden_assets'].iloc[0])})")
        prow = scores[scores.entity_id == pid].iloc[0]
        c = st.columns(3)
        c[0].metric("Declared income", pkr(prow.declared_income))
        c[1].metric("Hidden assets (associates)", pkr(prow.hidden_assets))
        c[2].metric("Household associates", int(prow.n_assoc))
        st.markdown("**Audit trail:**")
        for reason in json.loads(prow.audit_trail):
            st.markdown(f"- {reason}")
        st.markdown("**Network — 🎯 principal (red), 👤 associates/proxies (purple), "
                    "assets coloured by type:**")
        render_network(G, feats, pid, radius=3, height=520)

# ==========================================================================
with tab_live:
    st.subheader("🔎 Live entity match")
    st.markdown(
        "Type a raw record the way a messy registry would hold it — any spelling, "
        "**Urdu or Roman script**, partial address. The matcher blocks it against the "
        "resolved population and scores candidates with the **same unsupervised "
        "Fellegi-Sunter weights** the pipeline learned. No staging: invent an input "
        "on the spot.")
    with st.form("live_match"):
        a, b = st.columns(2)
        with a:
            q_name = st.text_input("Name (Urdu or Roman)", "Mohd Asif Khan")
            q_father = st.text_input("Father's name (optional)", "")
            q_city = st.text_input("City (optional)", "Lahore")
            q_dob = st.text_input("Date of birth YYYY-MM-DD (optional)", "")
        with b:
            q_cnic = st.text_input("CNIC (full or masked, optional)", "")
            q_phone = st.text_input("Phone (optional)", "")
            q_addr = st.text_input("Address (optional)", "")
        go = st.form_submit_button("🔎 Match against the population")

    if go:
        with st.spinner("Blocking + scoring candidates …"):
            idx = load_match_index()
            cands = live_match.match_record({
                "name": q_name, "father": q_father, "city": q_city, "dob": q_dob,
                "cnic": q_cnic, "phone": q_phone, "address": q_addr,
            }, index=idx)
        if not cands:
            st.warning("No candidate entities share a blocking key with this record.")
        else:
            st.caption(f"{len(cands)} candidate entit(y/ies); a high posterior needs a "
                       "hard identifier (CNIC / phone / address / DOB) — name alone is "
                       "deliberately weak.")
            for c in cands:
                srow = scores[scores.entity_id == c["entity_id"]]
                dev = f"{srow.deviation_score.iloc[0]:.0f}" if len(srow) else "n/a"
                with st.container(border=True):
                    cc1, cc2, cc3 = st.columns([2, 1, 1])
                    cc1.markdown(f"**Entity #{c['entity_id']}** — matched record "
                                 f"`{c['candidate_record']}` ({c['registry']}): "
                                 f"{c['candidate_name']}")
                    cc2.metric("Match posterior", f"{c['posterior']:.3f}")
                    cc3.metric("Deviation score", dev)
                    st.caption("Evidence: " + ", ".join(c["evidence"]))

# ==========================================================================
with tab_new:
    st.subheader("➕ Score a new individual")
    st.markdown("Enter someone's records and the **trained Isolation Forest** predicts "
                "their Tax Compliance Deviation Score and audit trail (live inference — "
                "no retraining).")
    with st.form("new_person"):
        a, b = st.columns(2)
        with a:
            st.markdown("**Declared / filing**")
            declared = st.number_input("Declared annual income (PKR)", 0, 1_000_000_000,
                                       0, step=100_000)
            is_filer = st.checkbox("Files a tax return", value=False)
            trips = st.number_input("International trips (last year)", 0, 50, 0)
        with b:
            st.markdown("**Observable footprint**")
            cc = st.number_input("Largest vehicle engine (cc)", 0, 6000, 0, step=100)
            veh_val = st.number_input("Total vehicle value (PKR)", 0, 1_000_000_000,
                                      0, step=500_000)
            prop_val = st.number_input("Total property value (PKR)", 0, 5_000_000_000,
                                       0, step=1_000_000)
            bill = st.number_input("Avg monthly electricity bill (PKR)", 0, 5_000_000,
                                   0, step=5_000)
        with st.expander("Advanced — household / proxy links (optional)"):
            net_assets = st.number_input(
                "Undeclared assets held by household associates (PKR)",
                0, 5_000_000_000, 0, step=1_000_000,
                help="Assets registered to non-filing relatives/proxies at the same address")
            net_n = st.number_input("Number of such associates", 0, 20, 0)
        submitted = st.form_submit_button("🔎 Score this individual")

    if submitted:
        result = score_person({
            "declared_income": declared, "is_filer": is_filer,
            "max_vehicle_cc": cc, "total_vehicle_value": veh_val,
            "total_property_value": prop_val, "max_monthly_bill": bill,
            "intl_trips": trips,
            "network_nonfiler_asset_value": net_assets, "network_neighbor_count": net_n,
        })
        sc = result["deviation_score"]
        verdict = ("🔴 HIGH — recommend audit" if sc >= 70 else
                   "🟠 MEDIUM — review" if sc >= 45 else
                   "🟢 LOW — consistent with declared income")
        m = st.columns(3)
        m[0].metric("Deviation Score", f"{sc:.0f}/100", verdict)
        m[1].metric("Declared income", pkr(result["declared_income"]))
        m[2].metric("Lifestyle-implied", f"~{pkr(result['implied_income'])}")
        # rank vs the analysed population
        pct = (scores.deviation_score < sc).mean() * 100
        st.altair_chart(chart_placement(scores, sc), use_container_width=True)
        st.caption(f"Red line = this individual. Scores higher than {pct:.0f}% of the "
                   f"{len(scores):,} analysed individuals.")
        st.markdown("**Why this score — audit trail:**")
        for reason in result["audit_trail"]:
            st.markdown(f"- {reason}")
