"""Unsupervised Fellegi-Sunter weighting.

No labels are used. The model calibrates itself from the data:

  m-probabilities (band frequency among true matches) are estimated from
  anchor pairs - records sharing an identical full 13-digit CNIC, which on
  civic data are matches with near-certainty.

  u-probabilities (band frequency among random non-matches) are estimated
  from randomly sampled record pairs with different CNICs.

Each feature band then carries an evidence weight log2(m/u). A pair's score
is the sum of weights of its observed bands; converted to a posterior match
probability using prior odds estimated from the candidate set.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

from .features import compare
from .records import Record

CLIP = 8.0
SMOOTH = 0.5


def anchor_pairs(records: list[Record], cap: int = 4000,
                 rng: random.Random | None = None) -> list[tuple[int, int]]:
    rng = rng or random.Random(0)
    by_cnic: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        if r.cnic["digits"]:
            by_cnic[r.cnic["digits"]].append(i)
    pairs = []
    for idxs in by_cnic.values():
        if 2 <= len(idxs) <= 8:
            for a in range(len(idxs) - 1):
                pairs.append((idxs[a], idxs[a + 1]))
    rng.shuffle(pairs)
    return pairs[:cap]


def random_nonmatch_pairs(records: list[Record], n: int = 30_000,
                          rng: random.Random | None = None) -> list[tuple[int, int]]:
    rng = rng or random.Random(1)
    out = []
    n_rec = len(records)
    while len(out) < n:
        i, j = rng.randrange(n_rec), rng.randrange(n_rec)
        if i == j:
            continue
        a, b = records[i], records[j]
        if a.cnic["digits"] and b.cnic["digits"] and a.cnic["digits"] == b.cnic["digits"]:
            continue
        out.append((min(i, j), max(i, j)))
    return out


class FellegiSunter:
    def __init__(self) -> None:
        self.weights: dict[str, float] = {}
        self.prior_logodds: float = -10.0

    def fit(self, records: list[Record], candidate_count: int,
            rng: random.Random | None = None,
            extra_matches: list[tuple[int, int]] | None = None,
            extra_nonmatches: list[tuple[int, int]] | None = None,
            feedback_weight: int = 50) -> dict:
        anchors = anchor_pairs(records, rng=rng)
        randoms = random_nonmatch_pairs(records, rng=rng)

        m_counts: Counter[str] = Counter()
        for i, j in anchors:
            # CNIC bands are excluded from anchor stats: anchors are selected
            # BY equal CNIC, so its m-probability there is 1 by construction.
            for band in compare(records[i], records[j]):
                m_counts[band] += 1
        u_counts: Counter[str] = Counter()
        for i, j in randoms:
            for band in compare(records[i], records[j]):
                u_counts[band] += 1

        n_m, n_u = max(len(anchors), 1), max(len(randoms), 1)

        # Human-in-the-loop active learning: each auditor decision enters as
        # `feedback_weight` pseudo-observations, so a handful of reviewed pairs
        # measurably move the weights without overwhelming the calibration.
        extra_matches = extra_matches or []
        extra_nonmatches = extra_nonmatches or []
        for i, j in extra_matches:
            for band in compare(records[i], records[j]):
                m_counts[band] += feedback_weight
            n_m += feedback_weight
        for i, j in extra_nonmatches:
            for band in compare(records[i], records[j]):
                u_counts[band] += feedback_weight
            n_u += feedback_weight
        bands = set(m_counts) | set(u_counts)
        for band in bands:
            m = (m_counts[band] + SMOOTH) / (n_m + SMOOTH * 2)
            u = (u_counts[band] + SMOOTH) / (n_u + SMOOTH * 2)
            w = math.log2(m / u)
            self.weights[band] = max(-CLIP, min(CLIP, w))

        # CNIC identity is itself decisive evidence; assign weights directly.
        self.weights["cnic_exact"] = CLIP
        self.weights["cnic_close"] = CLIP * 0.6
        self.weights["cnic_masked_match"] = 2.0
        self.weights["cnic_diff"] = -CLIP
        self.weights["cnic_diff_prefix"] = -2.0

        # DOB is the strongest non-CNIC disambiguator: exact birth-date agreement
        # across registries is near-unique among same-name people. Floor the
        # learned weights so a same-DOB borderline pair clears thresholds even
        # when the only other identifier is a masked CNIC.
        self.weights["dob_match"] = max(self.weights.get("dob_match", 0.0), 3.0)
        self.weights["dob_diff"] = min(self.weights.get("dob_diff", 0.0), -2.0)

        est_matches = max(len(anchors), 1)
        prior = min(0.5, est_matches / max(candidate_count, 1))
        self.prior_logodds = math.log2(prior / (1 - prior))
        return {
            "n_anchor_pairs": len(anchors),
            "n_random_pairs": len(randoms),
            "n_feedback_matches": len(extra_matches),
            "n_feedback_nonmatches": len(extra_nonmatches),
            "weights": dict(sorted(self.weights.items())),
            "prior_logodds": round(self.prior_logodds, 3),
        }

    def score(self, a: Record, b: Record) -> tuple[float, list[str]]:
        bands = compare(a, b)
        return sum(self.weights.get(band, 0.0) for band in bands), bands

    def posterior(self, score: float) -> float:
        logodds = self.prior_logodds + score
        if logodds > 50:
            return 1.0
        odds = 2.0 ** logodds
        return odds / (1 + odds)
