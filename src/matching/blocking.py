"""Blocking: cheap keys that put plausibly-matching records in the same bucket
so we never compare all O(n^2) pairs. A pair is a candidate if the two records
share at least one block key.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from .records import Record

# Skeletons of Muhammad-style prefixes carry no discriminating signal.
_PREFIX_SKELS = {"m", "mhmd", "mhd"}

MAX_BLOCK = 250


def block_keys(r: Record) -> list[str]:
    keys = []
    if r.cnic["digits"]:
        keys.append(f"C:{r.cnic['digits']}")
    if r.phone:
        keys.append(f"P:{r.phone}")
    skels = [s for s in r.name_skels if s not in _PREFIX_SKELS]
    if len(skels) >= 2:
        first, last = skels[0], skels[-1]
        keys.append(f"N:{first}|{last}")
        if r.city:
            keys.append(f"NC:{first}|{last}|{r.city}")
    elif len(skels) == 1 and r.city:
        keys.append(f"N1:{skels[0]}|{r.city}")
    if r.addr and r.addr["house"] is not None and r.addr["city"]:
        keys.append(f"A:{r.addr['city']}:{r.addr['house']}:{r.addr['street']}")
    # DOB co-blocks same-person records whose first name differs across script
    # (Urdu vs Roman) but whose date of birth agrees — the direct recall lift.
    # Pair it with the last-name skeleton to keep buckets tiny on a high-traffic
    # birthday, falling back to bare DOB when there is no usable name skeleton.
    if r.dob:
        if skels:
            keys.append(f"D:{r.dob}|{skels[-1]}")
        else:
            keys.append(f"D:{r.dob}")
    return keys


def candidate_pairs(records: list[Record]) -> tuple[set[tuple[int, int]], dict]:
    """Indices of candidate pairs plus blocking statistics."""
    blocks: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        for key in block_keys(r):
            blocks[key].append(i)

    pairs: set[tuple[int, int]] = set()
    skipped = 0
    for key, idxs in blocks.items():
        if len(idxs) > MAX_BLOCK:
            skipped += 1
            continue
        for i, j in combinations(sorted(set(idxs)), 2):
            pairs.add((i, j))
    stats = {
        "n_blocks": len(blocks),
        "skipped_oversize_blocks": skipped,
        "candidate_pairs": len(pairs),
    }
    return pairs, stats
