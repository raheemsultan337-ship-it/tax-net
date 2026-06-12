"""
Stage 2 — Entity Resolution.

Goal: link records scattered across the five observable silos into unified
"person entities", WITHOUT a clean shared key (CNICs are dirty; names drift
across Urdu/English transliterations).

Pipeline:
  1. Load + unify all observable records into one long table of "mentions".
  2. Normalize:
       - CNIC -> digits only (recover from formatting noise)
       - Name -> transliterate Urdu->Latin, canonicalize common variants,
                 strip honorifics, sort tokens -> a comparable key
  3. Block: candidate pairs only within blocks (CNIC prefix OR name phonetic
     key) so we never do all-vs-all.
  4. Match: score candidate pairs (CNIC similarity + name similarity).
  5. Cluster: connected components over high-confidence edges -> entities.

Reads ONLY data/observable/*.csv. Writes data/resolved/mentions.csv (each
observable record tagged with a predicted entity_id).
"""

import os
import re
import unicodedata
from collections import defaultdict

import pandas as pd
import networkx as nx
from rapidfuzz import fuzz
import jellyfish

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBS_DIR = os.path.join(ROOT, "data", "observable")
OUT_DIR = os.path.join(ROOT, "data", "resolved")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

# Urdu script -> Latin canonical token (covers the variants in the generator)
URDU_MAP = {
    "محمد": "muhammad", "احمد": "ahmad", "علی": "ali", "حسن": "hasan",
    "حسین": "husain", "عثمان": "usman", "بلال": "bilal", "حمزہ": "hamza",
    "عمران": "imran", "فیصل": "faisal", "عائشہ": "ayesha", "فاطمہ": "fatima",
    "زینب": "zainab", "مریم": "maryam", "ثناء": "sana", "حرا": "hira",
    "خان": "khan", "ملک": "malik", "شیخ": "sheikh", "چوہدری": "chaudhry",
    "بٹ": "butt", "قریشی": "qureshi", "سید": "syed", "اعوان": "awan",
    "راجہ": "raja", "مغل": "mughal",
}

# Latin transliteration variants -> canonical token
LATIN_CANON = {
    "mohammad": "muhammad", "mohammed": "muhammad", "muhammed": "muhammad",
    "ahmed": "ahmad",
    "hassan": "hasan",
    "hussain": "husain", "hussein": "husain",
    "osman": "usman", "othman": "usman",
    "hamzah": "hamza",
    "faysal": "faisal",
    "aisha": "ayesha", "aysha": "ayesha", "ayisha": "ayesha",
    "fatimah": "fatima", "fatema": "fatima",
    "zaynab": "zainab",
    "mariam": "maryam", "marium": "maryam",
    "malick": "malik",
    "shaikh": "sheikh", "shaykh": "sheikh",
    "chaudhary": "chaudhry", "choudhary": "chaudhry", "ch": "chaudhry",
    "bhatti": "butt",
    "quraishi": "qureshi",
    "sayed": "syed", "sayyid": "syed",
    "moghul": "mughal", "moghul": "mughal",
}

HONORIFICS = {"mr", "mrs", "ms", "dr", "syed"}  # syed also a name; handled gently


def normalize_cnic(raw):
    if not isinstance(raw, str):
        return ""
    digits = re.sub(r"\D", "", raw)
    return digits if len(digits) == 13 else ""


def _canon_token(tok):
    tok = tok.strip()
    if not tok:
        return ""
    if tok in URDU_MAP:
        return URDU_MAP[tok]
    # strip diacritics / punctuation from latin
    tok = unicodedata.normalize("NFKD", tok)
    tok = "".join(c for c in tok if not unicodedata.combining(c))
    tok = re.sub(r"[^a-zA-Z]", "", tok).lower()
    if not tok:
        return ""
    return LATIN_CANON.get(tok, tok)


def normalize_name(raw):
    """Return (canonical_string, sorted_token_tuple)."""
    if not isinstance(raw, str):
        return "", ()
    toks = [_canon_token(t) for t in raw.split()]
    toks = [t for t in toks if t and t not in HONORIFICS or t == "syed"]
    toks = [t for t in toks if t]
    # 'muhammad' as a prefix is extremely common filler -> keep but de-weight later
    sorted_toks = tuple(sorted(toks))
    return " ".join(toks), sorted_toks


def name_phonetic_key(sorted_toks):
    """Block key: metaphone of the rarest (non-muhammad) token, or first token."""
    toks = [t for t in sorted_toks if t != "muhammad"] or list(sorted_toks)
    if not toks:
        return "_"
    # use the alphabetically-last token's metaphone for a stable block
    return jellyfish.metaphone(toks[-1]) or toks[-1]


# ---------------------------------------------------------------------------
# Load all observable records into a unified "mentions" table
# ---------------------------------------------------------------------------
SCHEMAS = {
    "vehicles":    ("record_id", "owner_name", "owner_cnic"),
    "real_estate": ("record_id", "owner_name", "owner_cnic"),
    "utilities":   ("account_id", "consumer_name", "consumer_cnic"),
    "travel":      ("record_id", "passenger_name", "passport_cnic"),
    "tax_returns": ("return_id", "filer_name", "filer_cnic"),
}


def _row_city(src, r):
    """Best-effort city from whatever location field the source carries."""
    if src in ("vehicles", "real_estate"):
        return str(getattr(r, "city", "")).strip().lower()
    if src == "utilities":
        addr = str(getattr(r, "address", ""))
        return addr.split(",")[-1].strip().lower() if "," in addr else ""
    return ""  # travel, tax_returns carry no location


def load_mentions():
    rows = []
    for src, (idc, namec, cnicc) in SCHEMAS.items():
        df = pd.read_csv(os.path.join(OBS_DIR, f"{src}.csv"), keep_default_na=False)
        for r in df.itertuples():
            raw_name = getattr(r, namec)
            raw_cnic = getattr(r, cnicc)
            canon, toks = normalize_name(raw_name)
            _, father_toks = normalize_name(getattr(r, "father_name", ""))
            dob = str(getattr(r, "dob", "")).strip()
            rows.append({
                "record_id": getattr(r, idc),
                "source": src,
                "raw_name": raw_name,
                "norm_name": canon,
                "tokens": toks,
                "father_tokens": father_toks,
                "dob": dob,
                "cnic": normalize_cnic(raw_cnic),
                "city": _row_city(src, r),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Blocking + matching
# ---------------------------------------------------------------------------

def build_blocks(mentions):
    """Map block_key -> list of mention indices. Two blocking schemes unioned."""
    blocks = defaultdict(list)
    for i, m in mentions.iterrows():
        if m["cnic"]:
            blocks[f"cnic:{m['cnic'][:6]}"].append(i)        # CNIC prefix block
        blocks[f"name:{name_phonetic_key(m['tokens'])}"].append(i)  # phonetic block
    return blocks


def name_similarity(toks_a, toks_b):
    """Token-set similarity with 'muhammad' de-weighted (it's ubiquitous filler)."""
    a = set(toks_a)
    b = set(toks_b)
    if not a or not b:
        return 0.0
    # exact token overlap (Jaccard) ignoring the muhammad-only signal
    core_a = a - {"muhammad"} or a
    core_b = b - {"muhammad"} or b
    inter = core_a & core_b
    union = core_a | core_b
    jacc = len(inter) / len(union) if union else 0.0
    # fuzzy string backup for near-miss tokens
    fz = fuzz.token_set_ratio(" ".join(sorted(a)), " ".join(sorted(b))) / 100.0
    return max(jacc, 0.6 * jacc + 0.4 * fz)


def cnic_typo_match(c1, c2):
    """True if two 13-digit CNICs differ by at most one digit (a likely typo)."""
    if not c1 or not c2 or len(c1) != 13 or len(c2) != 13:
        return False
    return sum(a != b for a, b in zip(c1, c2)) == 1


ATTACH_THRESHOLD = 0.55   # min name sim to attach a CNIC-less record to a core
ATTACH_MARGIN = 0.15      # best core must beat 2nd-best by this (avoid ambiguity)


def resolve():
    """
    CNIC-authoritative clustering (avoids transitive name-bridge over-merging):
      1. Each distinct valid CNIC seeds an entity 'core'.
      2. Merge two cores whose CNICs are off-by-one-digit AND names agree (typo
         recovery) — does NOT chain, because we require direct name agreement.
      3. Attach each CNIC-less record to the single best-matching core in its
         name-phonetic block, only if above threshold AND unambiguous (margin).
         Otherwise it becomes its own singleton entity.
    """
    mentions = load_mentions()

    has_cnic = mentions[mentions["cnic"] != ""]
    no_cnic = mentions[mentions["cnic"] == ""]

    # --- 1. seed cores by exact CNIC -----------------------------------------
    core_of_cnic = {}          # cnic -> core_id
    core_tokens = {}           # core_id -> representative token set (union)
    core_dobs = defaultdict(set)   # core_id -> set of DOBs seen
    for cnic in has_cnic["cnic"].unique():
        cid = len(core_of_cnic)
        core_of_cnic[cnic] = cid
    entity_of = {}
    for idx, m in has_cnic.iterrows():
        cid = core_of_cnic[m["cnic"]]
        entity_of[idx] = cid
        core_tokens.setdefault(cid, set()).update(m["tokens"])
        if m["dob"]:
            core_dobs[cid].add(m["dob"])

    # --- 2. typo recovery: merge off-by-one CNIC cores with agreeing names ----
    # Block candidates by NAME phonetic key (not CNIC prefix): a typo can land
    # anywhere in the 13 digits, so a prefix block would miss ~half of them.
    uf = {cid: cid for cid in set(core_of_cnic.values())}
    def find(x):
        while uf[x] != x:
            uf[x] = uf[uf[x]]; x = uf[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: uf[max(ra, rb)] = min(ra, rb)

    cnic_of_core = {cid: c for c, cid in core_of_cnic.items()}
    # Blocking for typo recovery. Two complementary, high-cardinality keys keep
    # candidate blocks SMALL so this stays near-linear as the population grows:
    #   - DOB: a typo'd-CNIC record keeps the right DOB, so same-person cores
    #     share it. DOB has ~16k distinct values -> tiny blocks even at scale.
    #   - name token metaphone: fallback for cores whose records lack a DOB.
    # (With a realistic, high-cardinality name vocabulary the name blocks would
    # also stay small; the synthetic generator deliberately uses few names.)
    cores_by_dob = defaultdict(set)
    for cid, dobs in core_dobs.items():
        for d in dobs:
            cores_by_dob[d].add(cid)
    cores_by_tok = defaultdict(set)
    for cid, toks in core_tokens.items():
        for t in toks:
            if t != "muhammad":                       # too common to be discriminative
                cores_by_tok[jellyfish.metaphone(t) or t].add(cid)

    merges = 0
    seen_pairs = set()
    candidate_blocks = list(cores_by_dob.values()) + list(cores_by_tok.values())
    for cids in candidate_blocks:
        cids = list(cids)
        if len(cids) > 300:            # skip degenerate huge blocks (low-cardinality keys)
            continue
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                a, b = cids[i], cids[j]
                pk = (a, b) if a < b else (b, a)
                if pk in seen_pairs:
                    continue
                seen_pairs.add(pk)
                if cnic_typo_match(cnic_of_core[a], cnic_of_core[b]) and \
                   name_similarity(tuple(core_tokens.get(a, ())), tuple(core_tokens.get(b, ()))) >= 0.7:
                    union(a, b); merges += 1
    # relabel cores after typo merges
    for idx in list(entity_of):
        entity_of[idx] = find(entity_of[idx])
    # rebuild merged core token sets + city / dob / father aggregates
    merged_tokens = defaultdict(set)
    merged_cities = defaultdict(set)
    merged_dobs = defaultdict(set)
    merged_fathers = defaultdict(set)
    for idx, m in has_cnic.iterrows():
        e = entity_of[idx]
        merged_tokens[e].update(m["tokens"])
        if m["city"]:
            merged_cities[e].add(m["city"])
        if m["dob"]:
            merged_dobs[e].add(m["dob"])
        if m["father_tokens"]:
            merged_fathers[e].update(m["father_tokens"])

    # --- 3. attach CNIC-less records to best unambiguous core -----------------
    # Multi-key blocking: index merged cores by EVERY name token's metaphone, so
    # a true core is found even when the record's blocking key differs. The
    # strict margin below still rejects genuinely ambiguous (same-name) cases —
    # forcing those in was measured at ~0.16 precision, so we leave them split.
    cores_by_tok2 = defaultdict(set)
    for ent_id, toks in merged_tokens.items():
        for t in toks:
            if t != "muhammad":
                cores_by_tok2[jellyfish.metaphone(t) or t].add(ent_id)
    # DOB index for scalable, high-cardinality blocking of the attach step.
    cores_by_dob2 = defaultdict(set)
    for ent_id, dobs in merged_dobs.items():
        for d in dobs:
            cores_by_dob2[d].add(ent_id)

    def attach_score(m, c):
        """Name similarity + DOB / father's-name / city evidence. DOB and
        father's name separate same-named people: a matching core scores far
        above same-name decoys, so the margin below passes at high precision."""
        s = name_similarity(m["tokens"], tuple(merged_tokens[c]))
        if m["dob"] and merged_dobs.get(c):
            s += 0.6 if m["dob"] in merged_dobs[c] else -0.5        # exact DOB is near-unique
        if m["father_tokens"] and merged_fathers.get(c):
            fsim = name_similarity(m["father_tokens"], tuple(merged_fathers[c]))
            s += 0.4 * fsim - 0.2 * (1 - fsim)                     # +0.2 if match, -0.2 if not
        if m["city"] and m["city"] in merged_cities.get(c, set()):
            s += 0.15
        return s

    next_singleton = max(entity_of.values()) + 1 if entity_of else 0
    attached = 0
    for idx, m in no_cnic.iterrows():
        # DOB-first blocking keeps the candidate set tiny (near-linear at scale);
        # fall back to name-metaphone blocking only when the DOB is missing or
        # matches no core, so recall is preserved.
        cand = set()
        if m["dob"]:
            cand = set(cores_by_dob2.get(m["dob"], set()))
        if not cand:
            for t in m["tokens"]:
                if t != "muhammad":
                    cand |= cores_by_tok2.get(jellyfish.metaphone(t) or t, set())
        scored = sorted(((attach_score(m, c), c) for c in cand), reverse=True)
        if scored and scored[0][0] >= ATTACH_THRESHOLD and (
            len(scored) == 1 or scored[0][0] - scored[1][0] >= ATTACH_MARGIN
        ):
            entity_of[idx] = scored[0][1]
            attached += 1
        else:
            entity_of[idx] = next_singleton
            next_singleton += 1

    mentions["entity_id"] = mentions.index.map(entity_of)
    mentions_out = mentions.drop(columns=["tokens", "father_tokens"])
    mentions_out.to_csv(os.path.join(OUT_DIR, "mentions.csv"), index=False)

    n_entities = mentions["entity_id"].nunique()
    print(f"  mentions (records)         : {len(mentions)}")
    print(f"  records with usable CNIC   : {len(has_cnic)}  ({len(has_cnic)/len(mentions)*100:.0f}%)")
    print(f"  CNIC-less records          : {len(no_cnic)}  (attached: {attached}, singletons: {len(no_cnic)-attached})")
    print(f"  typo-CNIC core merges      : {merges}")
    print(f"  resolved entities          : {n_entities}")
    print(f"\n  wrote data/resolved/mentions.csv")
    return mentions


if __name__ == "__main__":
    resolve()
