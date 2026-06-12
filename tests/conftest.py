"""Shared pytest fixtures: run the pipeline once per session, expose outputs."""
import os
import sys
import pickle

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)
os.environ["PYTHONUTF8"] = "1"

DATA = os.path.join(ROOT, "data")
OBS = os.path.join(DATA, "observable")
RES = os.path.join(DATA, "resolved")
GT = os.path.join(DATA, "ground_truth")


@pytest.fixture(scope="session", autouse=True)
def pipeline():
    """Generate data + run the full pipeline once so all tests share fresh outputs."""
    import generate_data, entity_resolution, build_graph, scoring
    import gnn_detector, ensemble, rule_floors
    # Tests validate correctness, not scale — run at a small, fast, self-contained
    # population (decoupled from the production N_PERSONS) so the suite stays quick.
    # test_pipeline's entity-count bound reads generate_data.N_PERSONS, so it stays
    # consistent with whatever we set here.
    generate_data.N_PERSONS = 8000
    generate_data.main()
    entity_resolution.resolve()
    build_graph.build()
    scoring.score()
    gnn_detector.run()          # writes gnn_scores.csv (skips cleanly if torch absent)
    ensemble.combine()          # adds deviation_score_combined to entity_scores.csv
    rule_floors.compute_floors()  # entity_floors.csv + factors.json (explainable layer)
    return True


@pytest.fixture
def observable():
    return {f: pd.read_csv(os.path.join(OBS, f"{f}.csv"))
            for f in ["vehicles", "real_estate", "utilities", "travel", "tax_returns"]}


@pytest.fixture
def mentions():
    return pd.read_csv(os.path.join(RES, "mentions.csv"))


@pytest.fixture
def scores():
    return pd.read_csv(os.path.join(RES, "entity_scores.csv"))


@pytest.fixture
def features():
    return pd.read_csv(os.path.join(RES, "entity_features.csv"))


@pytest.fixture
def floors():
    return pd.read_csv(os.path.join(RES, "entity_floors.csv"))


@pytest.fixture
def graph():
    with open(os.path.join(RES, "graph.gpickle"), "rb") as fh:
        return pickle.load(fh)
