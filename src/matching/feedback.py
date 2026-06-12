"""Human-in-the-loop active learning store.

The review queue is where the cascade defers to a person. This module persists
those human decisions and feeds them back into Tier 1: each confirmed "same" /
"different" verdict re-enters the Fellegi-Sunter calibration as a batch of
weighted pseudo-observations, nudging the evidence weights toward what auditors
actually accept. The system learns from its auditors without ever becoming
supervised on the hidden ground truth (these are operator decisions, not the
answer key) — the wall to the sealed ground truth is untouched.
"""

from __future__ import annotations

import csv
from pathlib import Path

COLUMNS = ["record_a", "record_b", "label", "note"]
VALID_LABELS = {"same", "different"}


def _path(data_dir: str | Path) -> Path:
    return Path(data_dir) / "feedback" / "labeled_pairs.csv"


def load_feedback(data_dir: str | Path = "data") -> list[tuple[str, str, str]]:
    """Return [(record_a, record_b, label)] for every stored human decision."""
    p = _path(data_dir)
    if not p.exists():
        return []
    out: list[tuple[str, str, str]] = []
    with p.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            label = (row.get("label") or "").strip().lower()
            a, b = row.get("record_a"), row.get("record_b")
            if label in VALID_LABELS and a and b:
                out.append((a, b, label))
    return out


def append_decision(data_dir: str | Path, record_a: str, record_b: str,
                    label: str, note: str = "") -> None:
    """Persist one auditor decision. Applied on the next pipeline run."""
    label = label.strip().lower()
    if label not in VALID_LABELS:
        raise ValueError(f"label must be one of {VALID_LABELS}, got {label!r}")
    p = _path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_header = not p.exists()
    with p.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow({"record_a": record_a, "record_b": record_b,
                    "label": label, "note": note})
