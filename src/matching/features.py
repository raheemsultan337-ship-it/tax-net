"""Pairwise similarity features between two unified records.

Each feature maps to a small set of discrete bands so the Fellegi-Sunter
layer can attach evidence weights to them. Band value None means the
feature is unobservable for that pair (missing data) and contributes
nothing — absence of evidence is not evidence of absence.

Added over the donor: a DOB band. Exact date-of-birth agreement across
registries is near-unique among same-name people, so it is the strongest
non-CNIC disambiguator and the entity-resolution recall lever.
"""

from __future__ import annotations

from functools import lru_cache

from .records import Record


def jaro_winkler(a: str, b: str) -> float:
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if not la or not lb:
        return 0.0
    window = max(la, lb) // 2 - 1
    match_a = [False] * la
    match_b = [False] * lb
    matches = 0
    for i, ch in enumerate(a):
        lo = max(0, i - window)
        hi = min(lb, i + window + 1)
        for j in range(lo, hi):
            if not match_b[j] and b[j] == ch:
                match_a[i] = match_b[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0
    t = 0
    k = 0
    for i in range(la):
        if match_a[i]:
            while not match_b[k]:
                k += 1
            if a[i] != b[k]:
                t += 1
            k += 1
    jaro = (matches / la + matches / lb + (matches - t / 2) / matches) / 3
    prefix = 0
    for ca, cb in zip(a, b):
        if ca != cb or prefix == 4:
            break
        prefix += 1
    return jaro + prefix * 0.1 * (1 - jaro)


def damerau1(a: str, b: str) -> bool:
    """True if optimal-string-alignment distance <= 1 (one typo away)."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diffs = [i for i in range(la) if a[i] != b[i]]
        if len(diffs) == 1:
            return True
        if len(diffs) == 2 and diffs[1] == diffs[0] + 1:
            i = diffs[0]
            return a[i] == b[i + 1] and a[i + 1] == b[i]
        return False
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i = j = edits = 0
    while i < la and j < lb:
        if a[i] != b[j]:
            edits += 1
            if edits > 1:
                return False
            j += 1
        else:
            i += 1
            j += 1
    return True


def _token_set_sim(a: list[str] | tuple, b: list[str] | tuple) -> float:
    ta, tb = tuple(a), tuple(b)
    if ta > tb:
        ta, tb = tb, ta
    return _token_set_sim_cached(ta, tb)


@lru_cache(maxsize=2_000_000)
def _jw(a: str, b: str) -> float:
    return jaro_winkler(a, b)


@lru_cache(maxsize=2_000_000)
def _token_set_sim_cached(a: tuple, b: tuple) -> float:
    """Best-pairing average similarity between token tuples, order-free."""
    if not a or not b:
        return 0.0
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    total = 0.0
    used: set[int] = set()
    for s in short:
        best, best_j = 0.0, -1
        for j, l in enumerate(long_):
            if j in used:
                continue
            sim = _jw(s, l)
            if sim > best:
                best, best_j = sim, j
        total += best
        if best_j >= 0:
            used.add(best_j)
    return total / len(short)


def band_name(a: Record, b: Record) -> str | None:
    sim = _token_set_sim(a.name_skels, b.name_skels)
    if a.name_toks and b.name_toks:
        sim = max(sim, _token_set_sim(a.name_toks, b.name_toks))
    if sim >= 0.92:
        return "name_exact"
    if sim >= 0.78:
        return "name_close"
    if sim >= 0.60:
        return "name_weak"
    return "name_diff"


def band_father(a: Record, b: Record) -> str | None:
    if not a.father_skels or not b.father_skels:
        return None
    sim = _token_set_sim(a.father_skels, b.father_skels)
    if sim >= 0.85:
        return "father_match"
    if sim >= 0.6:
        return "father_weak"
    return "father_diff"


def band_cnic(a: Record, b: Record) -> str | None:
    ca, cb = a.cnic, b.cnic
    if ca["digits"] and cb["digits"]:
        if ca["digits"] == cb["digits"]:
            return "cnic_exact"
        if damerau1(ca["digits"], cb["digits"]):
            return "cnic_close"
        return "cnic_diff"
    pa, pb = ca["prefix"], cb["prefix"]
    if pa and pb:
        la, lb = ca["last"], cb["last"]
        if pa == pb and la and lb and la == lb:
            return "cnic_masked_match"
        if pa != pb:
            return "cnic_diff_prefix"
    return None


def band_dob(a: Record, b: Record) -> str | None:
    if not a.dob or not b.dob:
        return None
    return "dob_match" if a.dob == b.dob else "dob_diff"


def band_address(a: Record, b: Record) -> str | None:
    if not a.addr or not b.addr:
        return None
    aa, ab = a.addr, b.addr
    if aa["house"] is not None and ab["house"] is not None:
        same_no = aa["house"] == ab["house"] and (
            aa["street"] == ab["street"] or aa["street"] is None or ab["street"] is None)
        area_sim = _token_set_sim(aa["area_tokens"], ab["area_tokens"])
        if same_no and area_sim >= 0.6:
            return "addr_exact"
        if same_no and (aa["city"] == ab["city"] or not aa["city"] or not ab["city"]):
            return "addr_house_match"
    area_sim = _token_set_sim(aa["area_tokens"], ab["area_tokens"])
    if area_sim >= 0.75 and aa["city"] == ab["city"]:
        return "addr_area_match"
    return "addr_diff"


def band_phone(a: Record, b: Record) -> str | None:
    if not a.phone or not b.phone:
        return None
    return "phone_match" if a.phone == b.phone else "phone_diff"


def band_city(a: Record, b: Record) -> str | None:
    if not a.city or not b.city:
        return None
    return "city_match" if a.city == b.city else "city_diff"


FEATURES = [band_name, band_father, band_cnic, band_dob, band_address,
            band_phone, band_city]


def compare(a: Record, b: Record) -> list[str]:
    """All observable feature bands for a pair."""
    return [band for f in FEATURES if (band := f(a, b)) is not None]
