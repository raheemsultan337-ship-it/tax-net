"""Simplified Pakistani personal income tax slabs (salaried, FY2025-style).

Used by the rule-based lifestyle-floor cross-check (rule_floors.py) and the
dashboard to turn an estimated income into an expected tax. Mirrors the slab
table the data generator uses, so generated tax_paid and expected tax are on
the same footing.
"""

SLABS = [
    (600_000, 0.00),
    (1_200_000, 0.05),
    (2_200_000, 0.15),
    (3_200_000, 0.25),
    (4_100_000, 0.30),
    (float("inf"), 0.35),
]


def annual_tax(income: float) -> float:
    """Progressive tax due on an annual income in PKR."""
    tax = 0.0
    lower = 0.0
    for upper, rate in SLABS:
        if income <= lower:
            break
        taxable = min(income, upper) - lower
        tax += taxable * rate
        lower = upper
    return round(tax)
