"""Record-level corruption: typos, CNIC masking/typos, phone format variation."""

from __future__ import annotations

import random


def typo(s: str, rng: random.Random) -> str:
    """One character-level error: swap, drop, or duplicate. Skips Urdu text."""
    if len(s) < 4 or any("؀" <= ch <= "ۿ" for ch in s):
        return s
    i = rng.randint(1, len(s) - 2)
    kind = rng.random()
    if kind < 0.4:
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if kind < 0.7:
        return s[:i] + s[i + 1:]
    return s[:i] + s[i] + s[i:]


def maybe_typo(s: str, rng: random.Random, prob: float) -> str:
    return typo(s, rng) if rng.random() < prob else s


def make_cnic(rng: random.Random, city: str) -> str:
    """13-digit CNIC: 5-digit locality, 7-digit serial, 1 check digit."""
    prefixes = {
        "Islamabad": "61101", "Rawalpindi": "37405", "Lahore": "35202",
        "Karachi": "42101", "Peshawar": "17301", "Faisalabad": "33100",
        "Multan": "36302", "Sialkot": "34603", "Hyderabad": "41304",
        "Gujranwala": "34101", "Quetta": "54400", "Bahawalpur": "31202",
    }
    prefix = prefixes.get(city, "35202")
    serial = rng.randint(1_000_000, 9_999_999)
    check = rng.randint(1, 9)
    return f"{prefix}-{serial}-{check}"


def render_cnic(cnic: str, rng: random.Random, typo_prob: float = 0.02) -> str:
    """Format variation (with/without dashes) plus rare digit transposition."""
    digits = cnic.replace("-", "")
    if rng.random() < typo_prob:
        i = rng.randint(6, 10)
        digits = digits[:i] + digits[i + 1] + digits[i] + digits[i + 2:]
    if rng.random() < 0.25:
        return digits
    return f"{digits[:5]}-{digits[5:12]}-{digits[12]}"


def mask_cnic(cnic: str, rng: random.Random) -> str:
    digits = cnic.replace("-", "")
    if rng.random() < 0.5:
        return f"{digits[:5]}-XXXXXXX-{digits[12]}"
    return f"{digits[:5]}*****"


def make_phone(rng: random.Random) -> str:
    return f"03{rng.randint(0, 4)}{rng.randint(10_000_000, 99_999_999):08d}"


def render_phone(phone: str, rng: random.Random) -> str:
    digits = phone.replace("-", "").replace("+92", "0")
    r = rng.random()
    if r < 0.40:
        return digits
    if r < 0.70:
        return f"{digits[:4]}-{digits[4:]}"
    return f"+92{digits[1:]}"
