"""Bilingual audit-notice generation (JSON + Markdown + PDF)."""
import os
import json

import pytest

import audit_report
from conftest import ROOT

URDU_HEADER = "ٹیکس تعمیل جائزہ نوٹس"


def test_make_audits_writes_all_three_formats():
    pytest.importorskip("reportlab")
    res = audit_report.make_audits(top_n=3)
    assert res["audits_written"] == 3

    builder = audit_report.AuditBuilder()
    rank_col = ("deviation_score_combined" if "deviation_score_combined"
                in builder.scores.columns else "deviation_score")
    top = builder.scores.nlargest(3, rank_col)["entity_id"].tolist()
    audit_dir = os.path.join(ROOT, "data", "audit")
    for eid in top:
        for ext in ("json", "md", "pdf"):
            p = os.path.join(audit_dir, f"{eid}.{ext}")
            assert os.path.exists(p), f"missing {ext} for entity {eid}"
        assert os.path.getsize(os.path.join(audit_dir, f"{eid}.pdf")) > 0


def test_audit_markdown_is_bilingual_and_has_headline():
    builder = audit_report.AuditBuilder()
    rank_col = ("deviation_score_combined" if "deviation_score_combined"
                in builder.scores.columns else "deviation_score")
    eid = int(builder.scores.nlargest(1, rank_col)["entity_id"].iloc[0])
    audit = builder.build(eid)
    md = builder.markdown(audit)
    assert URDU_HEADER in md                      # bilingual header
    assert "Deviation Score" in md
    # the headline number is the ensemble score; ML reasons present
    assert audit["ml"]["deviation_score"] >= 0
    assert "cross_check" in audit


def test_audit_dict_carries_ml_and_crosscheck():
    builder = audit_report.AuditBuilder()
    eid = int(builder.scores["entity_id"].iloc[0])
    audit = builder.build(eid)
    assert set(audit["ml"]) >= {"deviation_score", "deviation_score_if",
                                "gnn_score", "reasons"}
    assert "tax_gap_pkr" in audit["cross_check"]
