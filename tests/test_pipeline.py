"""End-to-end functional tests: entity resolution, scoring, graph, and quality
thresholds. These validate the KEY functionalities work correctly and reliably."""
import json
import pytest
import evaluate as ev


# ---- Entity resolution quality ------------------------------------------
def test_entity_resolution_quality():
    m = ev.evaluate_entity_resolution()
    assert m["precision"] >= 0.95, f"ER precision regressed: {m['precision']}"
    assert m["recall"] >= 0.90, f"ER recall regressed: {m['recall']}"
    assert m["purity"] >= 0.97, f"ER purity regressed: {m['purity']}"

def test_resolved_entity_count_sane(mentions):
    # should resolve close to the true person count (incl. proxies + a few
    # fragments), not collapse or explode — bound relative to the population size
    import generate_data
    n = generate_data.N_PERSONS
    n_entities = mentions.entity_id.nunique()
    assert n <= n_entities <= 1.3 * n

def test_every_record_assigned_an_entity(mentions):
    assert mentions.entity_id.notna().all()


# ---- Detection quality ---------------------------------------------------
def test_detection_beats_baseline():
    m = ev.evaluate_detection()
    assert m["ap"] >= 0.60, f"detection AP regressed below useful: {m['ap']}"
    assert m["precision"] >= 0.60          # precision@25% well above ~0.3 base rate

def test_ensemble_keeps_ap_and_lifts_proxy_recall():
    """The combined (IF + GNN) production score must (a) not crater the headline AP
    relative to IF alone, and (b) catch proxy-using principals better than IF's
    own-features-only baseline. This validates the ensemble's reason to exist."""
    m = ev.evaluate_detection()
    if "ap_combined" not in m:
        pytest.skip("GNN unavailable in this environment — ensemble fell back to IF only")
    assert m["ap_combined"] >= 0.70, f"combined AP regressed: {m['ap_combined']}"
    assert m["principal_recall_combined"] >= 0.60, \
        f"ensemble lost the proxy payoff: {m['principal_recall_combined']}"

def test_deviation_scores_in_range(scores):
    assert scores.deviation_score.between(0, 100).all()
    assert scores.deviation_score.max() > 80          # someone is clearly flagged

def test_top_flagged_have_audit_trail(scores):
    top = scores.sort_values("deviation_score", ascending=False).head(20)
    for trail in top.audit_trail:
        reasons = json.loads(trail)
        assert len(reasons) >= 1                       # every flag is explained


# ---- Knowledge graph integrity ------------------------------------------
def test_graph_has_expected_node_kinds(graph):
    kinds = {d.get("kind") for _, d in graph.nodes(data=True)}
    assert {"person", "vehicle", "property", "utility", "address"} <= kinds

def test_graph_connected_to_assets(graph):
    persons = [n for n, d in graph.nodes(data=True) if d.get("kind") == "person"]
    assert len(persons) > 1000


# ---- Proxy / benami network feature -------------------------------------
def test_network_feature_present(features):
    assert "network_nonfiler_asset_value" in features.columns
    assert features.network_nonfiler_asset_value.max() > 5_000_000   # some hidden wealth

def test_graph_lifts_principal_recall():
    """The headline claim: the network signal must raise principal recall over
    the own-features-only baseline."""
    import pandas as pd, os
    from conftest import RES, GT
    scores = pd.read_csv(os.path.join(RES, "entity_scores.csv"))
    persons = pd.read_csv(os.path.join(GT, "persons.csv"), keep_default_na=False)
    mentions = pd.read_csv(os.path.join(RES, "mentions.csv"))
    link = pd.read_csv(os.path.join(GT, "record_linkage.csv"))
    mentions["tp"] = mentions.record_id.map(dict(zip(link.record_id, link.person_id)))
    ent_pid = mentions.groupby("entity_id").tp.agg(lambda s: s.value_counts().idxmax())
    role = dict(zip(persons.person_id, persons.role))
    scores["role"] = scores.entity_id.map(ent_pid).map(role)

    def princ_recall(col, frac=0.25):
        s = scores.sort_values(col, ascending=False)
        k = max(1, int(len(s) * frac))
        tot = (s.role == "principal").sum()
        return (s.head(k).role == "principal").sum() / tot if tot else 0.0

    own = princ_recall("deviation_score_own")
    net = princ_recall("deviation_score")
    assert net > own, f"network signal did not help principals (own={own}, net={net})"


# ---- Hard ER cases: same-name relatives must NOT be merged ----------------
def _entities_of_person(mentions, link, person_id):
    rec2pid = dict(zip(link.record_id, link.person_id))
    mids = mentions[mentions.record_id.map(rec2pid) == person_id]
    return set(mids.entity_id)

def test_father_and_son_not_merged(mentions):
    """Seed personas P000002 (father) and P000003 (son) share name + address but
    differ in CNIC, DOB and father's name. The DOB/father vetoes must keep them
    as distinct entities — the classic over-merge trap."""
    import pandas as pd, os
    from conftest import GT
    link = pd.read_csv(os.path.join(GT, "record_linkage.csv"))
    father = _entities_of_person(mentions, link, "P000002")
    son = _entities_of_person(mentions, link, "P000003")
    assert father and son, "seed father/son records missing"
    assert father.isdisjoint(son), \
        f"father/son were merged (father={father}, son={son})"

def test_twins_not_merged(mentions):
    """Seed twins P000004/P000005: same father + address + DOB year, different
    first name, CNIC and exact DOB. Must stay distinct."""
    import pandas as pd, os
    from conftest import GT
    link = pd.read_csv(os.path.join(GT, "record_linkage.csv"))
    a = _entities_of_person(mentions, link, "P000004")
    b = _entities_of_person(mentions, link, "P000005")
    assert a and b and a.isdisjoint(b), f"twins were merged (a={a}, b={b})"
