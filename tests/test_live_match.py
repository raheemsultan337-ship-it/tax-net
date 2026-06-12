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


def test_query_with_a_real_name_finds_its_entity():
    """Querying a multi-token Roman name that exists in the data must surface at
    least one candidate (the record itself shares a blocking key)."""
    import os
    import pandas as pd
    from conftest import RES
    mentions = pd.read_csv(os.path.join(RES, "mentions.csv"))
    sample = next(n for n in mentions.raw_name.dropna()
                  if str(n).isascii() and len(str(n).split()) >= 2)
    idx = live_match.load_index()
    res = live_match.match_record({"name": sample}, index=idx)
    assert len(res) >= 1
