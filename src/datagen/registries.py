"""Render the population into tax-net's five observable registry CSVs plus the
sealed ground-truth linkage.

Column names follow tax-net's existing observable schema (so build_graph and the
detector are unchanged); the rendering layer is the donor's (per-registry Urdu,
masked/typo CNICs, phone + address variation, real vehicle makes). Every record
carries DOB + father's name — the entity-resolution recall lever.

Registry files NEVER contain person_id; linking records back to people is the
ER pipeline's job. The (record_id, person_id, source) linkage is written
separately and read only by the evaluation layer.
"""

from __future__ import annotations

import random

import pandas as pd

from .tax import annual_tax
from .names import render_name, render_roman
from .addresses import render_address
from .noise import maybe_typo, mask_cnic, render_cnic, render_phone
from .personas import Person


def _name(p: Person, rng: random.Random, urdu_prob: float, typo_prob: float) -> str:
    return maybe_typo(render_name(p.name_tokens, rng, p.gender, urdu_prob), rng, typo_prob)


def _father(p: Person, rng: random.Random, typo_prob: float, missing_prob: float = 0.15) -> str:
    if rng.random() < missing_prob:
        return ""
    return maybe_typo(render_roman(p.father_tokens, rng, "M"), rng, typo_prob)


def _rec_dob(p: Person, rng: random.Random, missing_prob: float = 0.10) -> str:
    return "" if rng.random() < missing_prob else p.dob


def _cnic(p: Person, rng: random.Random, full_p: float, mask_p: float,
          cnic_typo: float) -> str:
    """Render a CNIC: full (with typo/format variation), masked, or omitted."""
    r = rng.random()
    if r < full_p:
        return render_cnic(p.cnic, rng, cnic_typo)
    if r < full_p + mask_p:
        return mask_cnic(p.cnic, rng)
    return ""


def build_vehicles(persons, rng, links, urdu_prob, typo_prob, cnic_typo):
    rows = []
    rid = 0
    for p in persons:
        for make, cc, year in p.vehicles:
            record_id = f"VEH{rid:06d}"
            reg_value = int(cc * rng.uniform(3200, 4200))
            rows.append({
                "record_id": record_id,
                "owner_name": _name(p, rng, urdu_prob, typo_prob),
                "owner_cnic": _cnic(p, rng, 0.70, 0.15, cnic_typo),
                "father_name": _father(p, rng, typo_prob),
                "dob": _rec_dob(p, rng),
                "engine_cc": cc,
                "reg_value": reg_value,
                "phone": render_phone(rng.choice(p.phones), rng) if rng.random() < 0.5 else "",
                "city": p.address.city,
            })
            links.append({"record_id": record_id, "person_id": p.person_id, "source": "vehicles"})
            rid += 1
    return pd.DataFrame(rows)


def build_real_estate(persons, rng, links, urdu_prob, typo_prob, cnic_typo):
    rows = []
    rid = 0
    for p in persons:
        for dc_value, marla in p.properties:
            record_id = f"PROP{rid:06d}"
            rows.append({
                "record_id": record_id,
                "owner_name": _name(p, rng, urdu_prob, typo_prob),
                "owner_cnic": _cnic(p, rng, 0.80, 0.0, cnic_typo),
                "father_name": _father(p, rng, typo_prob),
                "dob": _rec_dob(p, rng),
                "property_value": int(dc_value),
                # Stable canonical household string: the shared-address node that
                # links a principal to a proxy depends on byte-identical strings.
                "household": p.address.canonical(),
                "area": p.address.area,
                "city": p.address.city,
            })
            links.append({"record_id": record_id, "person_id": p.person_id, "source": "real_estate"})
            rid += 1
    return pd.DataFrame(rows)


def build_utilities(persons, rng, links, urdu_prob, typo_prob, cnic_typo):
    rows = []
    rid = 0
    for p in persons:
        if p.household_head is not None or p.avg_monthly_bill <= 0:
            continue
        account_id = f"UTL{rid:06d}"
        rows.append({
            "account_id": account_id,
            "consumer_name": _name(p, rng, urdu_prob, typo_prob),
            "consumer_cnic": _cnic(p, rng, 0.55, 0.15, cnic_typo),
            "father_name": _father(p, rng, typo_prob),
            "dob": _rec_dob(p, rng),
            "address": p.address.canonical(),
            "avg_monthly_bill": int(p.avg_monthly_bill),
            "phone": render_phone(rng.choice(p.phones), rng) if rng.random() < 0.4 else "",
            "city": p.address.city,
        })
        links.append({"record_id": account_id, "person_id": p.person_id, "source": "utilities"})
        rid += 1
    return pd.DataFrame(rows)


def build_travel(persons, rng, links, typo_prob, cnic_typo):
    rows = []
    rid = 0
    for p in persons:
        if p.trips_abroad <= 0:
            continue
        record_id = f"TRV{rid:06d}"
        rows.append({
            "record_id": record_id,
            "passenger_name": maybe_typo(render_roman(p.name_tokens, rng, p.gender), rng, typo_prob),
            "passport_cnic": render_cnic(p.cnic, rng, cnic_typo) if rng.random() < 0.55 else "",
            "father_name": _father(p, rng, typo_prob),
            "dob": _rec_dob(p, rng),
            "trips_last_year": int(p.trips_abroad),
            "destinations": "; ".join(p.destinations),
            "city": p.address.city,
        })
        links.append({"record_id": record_id, "person_id": p.person_id, "source": "travel"})
        rid += 1
    return pd.DataFrame(rows)


def build_tax_returns(persons, rng, links, cnic_typo):
    rows = []
    rid = 0
    for p in persons:
        if p.filer_status == "absent":
            continue
        return_id = f"TAX{rid:06d}"
        declared = p.declared_income
        rows.append({
            "return_id": return_id,
            "filer_name": render_roman(p.name_tokens, rng, p.gender),  # FBR uses official spelling
            "filer_cnic": render_cnic(p.cnic, rng, cnic_typo) if rng.random() < 0.95 else "",
            "father_name": " ".join(p.father_tokens),
            "dob": _rec_dob(p, rng, missing_prob=0.05),
            "declared_income": int(declared),
            "tax_paid": int(annual_tax(declared)),
            "city": p.address.city,
        })
        links.append({"record_id": return_id, "person_id": p.person_id, "source": "tax_returns"})
        rid += 1
    return pd.DataFrame(rows)
