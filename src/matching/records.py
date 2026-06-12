"""Load tax-net's five observable registries into one unified record schema.

Adapted from the donor (which read six `registries/` files) to tax-net's
`observable/` layout and column names, and extended with a `dob` field — the
DOB evidence band is tax-net's highest-signal disambiguator for same-name people.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .normalize import (
    is_urdu, name_skeletons, name_tokens, normalize_address, normalize_cnic,
    normalize_phone,
)

# registry -> (id_col, name_col, father_col, address_col, cnic_col, phone_col, dob_col)
SCHEMA = {
    "vehicles": ("record_id", "owner_name", "father_name", None,
                 "owner_cnic", "phone", "dob"),
    "real_estate": ("record_id", "owner_name", "father_name", "household",
                    "owner_cnic", None, "dob"),
    "utilities": ("account_id", "consumer_name", "father_name", "address",
                  "consumer_cnic", "phone", "dob"),
    "travel": ("record_id", "passenger_name", "father_name", None,
               "passport_cnic", None, "dob"),
    "tax_returns": ("return_id", "filer_name", "father_name", None,
                    "filer_cnic", None, "dob"),
}


@dataclass
class Record:
    record_id: str
    registry: str
    raw_name: str
    name_toks: list[str]
    name_skels: list[str]
    urdu: bool
    father_skels: list[str]
    addr: dict | None
    cnic: dict
    phone: str | None
    dob: str | None
    city: str | None
    raw: dict = field(default_factory=dict)


def _clean(v) -> str:
    s = str(v).strip()
    return "" if s.lower() == "nan" else s


def load_records(data_dir: str | Path) -> list[Record]:
    reg_dir = Path(data_dir) / "observable"
    records: list[Record] = []
    for registry, (id_c, name_c, father_c, addr_c, cnic_c, phone_c, dob_c) in SCHEMA.items():
        df = pd.read_csv(reg_dir / f"{registry}.csv", dtype=str)
        for row in df.to_dict("records"):
            name = _clean(row.get(name_c, ""))
            father = _clean(row.get(father_c, "")) if father_c else ""
            addr_s = _clean(row.get(addr_c, "")) if addr_c else ""
            dob = _clean(row.get(dob_c, "")) if dob_c else ""
            records.append(Record(
                record_id=row[id_c],
                registry=registry,
                raw_name=name,
                name_toks=name_tokens(name) if not is_urdu(name) else [],
                name_skels=name_skeletons(name),
                urdu=is_urdu(name),
                father_skels=name_skeletons(father) if father else [],
                addr=normalize_address(addr_s) if addr_s else None,
                cnic=normalize_cnic(_clean(row.get(cnic_c, "")) if cnic_c else ""),
                phone=normalize_phone(_clean(row.get(phone_c, "")) if phone_c else ""),
                dob=dob or None,
                city=_clean(row.get("city", "")).lower() or None,
                raw=row,
            ))
    return records
