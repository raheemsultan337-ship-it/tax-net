"""The WALL: the unsupervised detector must NEVER see ground truth. These tests
enforce the separation structurally — a core correctness + integrity guarantee."""
import os
import glob
import pandas as pd

from conftest import OBS, SRC

# columns that would leak the answer if they appeared in observable data
LEAKY_COLUMNS = {"person_id", "true_income", "asset_income", "report_ratio",
                 "is_evader", "non_filer", "role", "principal_id", "uses_proxy"}

# Modules that make up the unsupervised detector — none may reference ground truth.
# (generate_data.py / datagen/* legitimately WRITE ground truth, so they are not
#  detector modules and are excluded.)
DETECTOR_MODULES = ["entity_resolution.py", "build_graph.py", "scoring.py",
                    "gnn_detector.py", "ensemble.py", "score_person.py",
                    "rule_floors.py", "tax_slabs.py", "audit_report.py",
                    "live_match.py"]


def _detector_sources():
    paths = [os.path.join(SRC, m) for m in DETECTOR_MODULES]
    paths += glob.glob(os.path.join(SRC, "matching", "*.py"))
    return [p for p in paths if os.path.exists(p)]


def test_observable_data_has_no_leaky_columns():
    for f in ["vehicles", "real_estate", "utilities", "travel", "tax_returns"]:
        cols = set(pd.read_csv(os.path.join(OBS, f"{f}.csv"), nrows=1).columns)
        leaked = cols & LEAKY_COLUMNS
        assert not leaked, f"{f}.csv leaks ground-truth columns: {leaked}"


def test_detector_source_never_reads_ground_truth():
    for path in _detector_sources():
        src = open(path, encoding="utf-8").read()
        assert "ground_truth" not in src, \
            f"{os.path.relpath(path, SRC)} references ground_truth — the wall is breached"


def test_observable_records_have_no_person_id(observable):
    for name, df in observable.items():
        assert "person_id" not in df.columns, f"{name} exposes the true person_id"
