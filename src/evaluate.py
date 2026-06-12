"""
evaluate.py — THE ONLY module that reads ground truth.

It scores how well the (unsupervised) pipeline did, AFTER detection has run:
  - Entity Resolution: pairwise precision / recall / F1 + cluster purity
  - Detection: how well the Tax Compliance Deviation Score ranks/flags the
    independently-generated under-reporters (precision/recall/PR-AUC).

Nothing here feeds back into the detector — it is a scorecard only.
"""

import os
import itertools
from collections import defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_DIR = os.path.join(ROOT, "data", "ground_truth")
RES_DIR = os.path.join(ROOT, "data", "resolved")


def _pairs_by(df, col):
    groups = defaultdict(list)
    for idx, key in df[col].items():
        groups[key].append(idx)
    pairs = set()
    for idxs in groups.values():
        for a, b in itertools.combinations(sorted(idxs), 2):
            pairs.add((a, b))
    return pairs


def evaluate_entity_resolution():
    mentions = pd.read_csv(os.path.join(RES_DIR, "mentions.csv"))
    link = pd.read_csv(os.path.join(GT_DIR, "record_linkage.csv"))
    truth = dict(zip(link.record_id, link.person_id))
    mentions["true_pid"] = mentions.record_id.map(truth)

    pred = _pairs_by(mentions, "entity_id")
    true = _pairs_by(mentions, "true_pid")
    tp = len(pred & true)
    fp = len(pred - true)
    fn = len(true - pred)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    purity = (
        mentions.groupby("entity_id").true_pid
        .apply(lambda s: s.value_counts().iloc[0] / len(s))
        .mean()
    )
    print("=== Entity Resolution ===")
    print(f"  pairwise precision : {P:.3f}")
    print(f"  pairwise recall    : {R:.3f}")
    print(f"  pairwise F1        : {F:.3f}")
    print(f"  mean cluster purity: {purity:.3f}")
    print(f"  predicted entities : {mentions.entity_id.nunique()}  (true persons: {mentions.true_pid.nunique()})")
    return {"precision": P, "recall": R, "f1": F, "purity": purity}


def evaluate_detection(scores_path=None, top_k_frac=0.25):
    """Compare the deviation score against the hidden is_evader label."""
    scores_path = scores_path or os.path.join(RES_DIR, "entity_scores.csv")
    if not os.path.exists(scores_path):
        print("\n(detection scores not generated yet — run scoring.py first)")
        return None

    scores = pd.read_csv(scores_path)              # entity_id, deviation_score(s)  (NO ground truth)
    persons = pd.read_csv(os.path.join(GT_DIR, "persons.csv"), keep_default_na=False)
    role = dict(zip(persons.person_id, persons.get("role", "normal")))
    evader = dict(zip(persons.person_id, persons.is_evader))

    # Derive entity -> true person here (evaluate is allowed to read the linkage);
    # the detector never had this. Majority-vote the true person of each entity.
    mentions = pd.read_csv(os.path.join(RES_DIR, "mentions.csv"))
    link = pd.read_csv(os.path.join(GT_DIR, "record_linkage.csv"))
    mentions["true_pid"] = mentions.record_id.map(dict(zip(link.record_id, link.person_id)))
    ent_pid = mentions.groupby("entity_id").true_pid.agg(
        lambda s: s.value_counts().idxmax()
    )
    scores["true_pid"] = scores["entity_id"].map(ent_pid)
    scores["role"] = scores["true_pid"].map(role).fillna("normal")
    # audit-worthy = anyone hiding tax: self-evaders, proxy-using principals,
    # AND the proxies themselves (they hold unexplained wealth).
    scores["target"] = scores["true_pid"].map(evader).fillna(False) | (scores["role"] == "proxy")

    def _ap(col):
        s = scores.sort_values(col, ascending=False)
        y = s.target.astype(int).tolist()
        c = a = 0
        n = sum(y)
        for i, v in enumerate(y, 1):
            if v:
                c += 1
                a += c / i
        return a / n if n else 0.0

    def _principal_recall(col, frac):
        s = scores.sort_values(col, ascending=False)
        k = max(1, int(len(s) * frac))
        princ = s[s.role == "principal"]
        if not len(princ):
            return None
        return (s.head(k).role == "principal").sum() / len(princ)

    s = scores.sort_values("deviation_score", ascending=False).reset_index(drop=True)
    k = max(1, int(len(s) * top_k_frac))
    flagged = s.head(k)
    tp = int(flagged.target.sum())
    P = tp / k
    R = tp / int(s.target.sum()) if s.target.sum() else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    ap = _ap("deviation_score")

    print(f"\n=== Detection (top {int(top_k_frac*100)}% flagged as audit targets) ===")
    print(f"  audit-worthy targets: {int(s.target.sum())} ({s.target.mean()*100:.1f}% base rate)")
    print(f"  flagged entities   : {k}")
    print(f"  precision@{int(top_k_frac*100)}%      : {P:.3f}")
    print(f"  recall@{int(top_k_frac*100)}%         : {R:.3f}")
    print(f"  F1                 : {F:.3f}")
    print(f"  average precision  : {ap:.3f}  (rank quality, 1.0 = perfect)")

    # The graph demonstration: own-only vs own+network on proxy-using principals.
    if "deviation_score_own" in scores.columns:
        n_princ = int((scores.role == "principal").sum())
        pr_own = _principal_recall("deviation_score_own", top_k_frac)
        pr_net = _principal_recall("deviation_score", top_k_frac)
        print(f"\n  --- Graph contribution ({n_princ} proxy-using principals) ---")
        print(f"  overall AP  own-only : {_ap('deviation_score_own'):.3f}")
        print(f"  overall AP  +network : {ap:.3f}")
        if pr_own is not None:
            print(f"  principal recall@{int(top_k_frac*100)}%  own-only : {pr_own:.3f}")
            print(f"  principal recall@{int(top_k_frac*100)}%  +network : {pr_net:.3f}")

    # The ENSEMBLE: one production score blending IF (tabular) + GNN (relational).
    result = {"precision": P, "recall": R, "f1": F, "ap": ap}
    if "deviation_score_combined" in scores.columns and scores["deviation_score_combined"].notna().any():
        ap_c = _ap("deviation_score_combined")
        pr_c = _principal_recall("deviation_score_combined", top_k_frac)
        print(f"\n  --- Ensemble (IF + GNN -> one production score) ---")
        print(f"  combined AP                : {ap_c:.3f}")
        if pr_c is not None:
            print(f"  combined principal recall@{int(top_k_frac*100)}% : {pr_c:.3f}")
        result["ap_combined"] = ap_c
        result["principal_recall_combined"] = pr_c
    return result


if __name__ == "__main__":
    evaluate_entity_resolution()
    evaluate_detection()
