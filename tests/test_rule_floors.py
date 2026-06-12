"""Rule-based lifestyle-floor cross-check (secondary explainability layer)."""
import os
import json

import pandas as pd

import tax_slabs
from conftest import RES


# ---- Progressive tax slabs ------------------------------------------------
def test_annual_tax_below_threshold_is_zero():
    assert tax_slabs.annual_tax(600_000) == 0

def test_annual_tax_first_bracket():
    # 5% on the slice from 600k to 1.2M
    assert tax_slabs.annual_tax(1_200_000) == 30_000

def test_annual_tax_monotonic():
    vals = [tax_slabs.annual_tax(x) for x in (0, 1e6, 3e6, 1e7, 5e7)]
    assert vals == sorted(vals)


# ---- entity_floors.csv schema + invariants -------------------------------
def test_floors_schema(floors):
    expected = {"entity_id", "estimated_income_pkr", "declared_income_pkr",
                "filer_status", "tax_paid_pkr", "expected_tax_pkr", "tax_gap_pkr",
                "rule_deviation_score", "band"}
    assert expected <= set(floors.columns)
    assert floors.rule_deviation_score.between(0, 100).all()

def test_factors_json_shape():
    with open(os.path.join(RES, "factors.json"), encoding="utf-8") as fh:
        fac = json.load(fh)
    assert isinstance(fac, dict) and fac
    sample = next(iter(fac.values()))
    assert "factors" in sample and "notes" in sample

def test_high_band_has_a_tax_gap(floors):
    high = floors[floors.band == "high"]
    assert len(high) > 0
    assert (high.tax_gap_pkr > 0).all()

def test_false_positive_guard_compliant_filers(floors):
    """A filer whose declared income already covers the lifestyle floor has no
    gap — wealthy-but-compliant people are not flagged by the rule layer."""
    compliant = floors[(floors.filer_status == "filer")
                       & (floors.declared_income_pkr >= floors.estimated_income_pkr)]
    assert len(compliant) > 0, "expected some compliant filers in the population"
    assert (compliant.tax_gap_pkr == 0).all()
