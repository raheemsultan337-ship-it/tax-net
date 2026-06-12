"""Live single-record matching against the resolved population (smoke test)."""
import live_match


def test_match_record_returns_ranked_candidates():
    idx = live_match.load_index()
    res = live_match.match_record(
        {"name": "Mohd Asif Khan", "city": "Lahore"}, index=idx)
    assert isinstance(res, list)
    for c in res:
        assert isinstance(c["entity_id"], int)
        assert 0.0 <= c["posterior"] <= 1.0
        assert isinstance(c["evidence"], list)
    # posteriors are sorted high -> low
    posts = [c["posterior"] for c in res]
    assert posts == sorted(posts, reverse=True)


def test_cross_script_query_finds_a_candidate():
    """A Roman query should still reach Urdu-script records via the consonant
    skeleton blocking (and vice versa) — at least one candidate surfaces."""
    idx = live_match.load_index()
    res = live_match.match_record(
        {"name": "Muhammad Khan", "city": "Lahore"}, index=idx)
    assert len(res) >= 1
