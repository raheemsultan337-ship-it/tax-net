"""Script-independent normalization for Pakistani names and addresses.

The core trick is a consonant skeleton: both Roman spellings and Urdu script
collapse onto the same Latin consonant classes, so "Muhammad", "Mohammad",
"Mohd" and the Urdu rendering all reduce to (nearly) the same key without any
dictionary of known variants. Vowels carry almost no signal across
Urdu-to-Roman transliteration; consonants carry almost all of it.
"""

from __future__ import annotations

import re

HONORIFICS = {"mr", "mrs", "ms", "dr", "haji", "mian", "engr", "mst",
              "sahib", "sahiba", "prof", "syed", "syeda"}

# Roman digraphs that represent one Urdu consonant.
_DIGRAPHS = [("kh", "k"), ("gh", "g"), ("sh", "x"), ("ch", "c"), ("zh", "z"),
             ("th", "t"), ("dh", "d"), ("ph", "f"), ("bh", "b"), ("rh", "r"),
             ("wh", "v")]

# Urdu letter -> consonant class. Vowel carriers and diacritics map to "".
_URDU = {
    "ب": "b", "پ": "p", "ت": "t", "ٹ": "t", "ث": "s", "ج": "j", "چ": "c",
    "ح": "h", "خ": "k", "د": "d", "ڈ": "d", "ذ": "z", "ر": "r", "ڑ": "r",
    "ز": "z", "ژ": "z", "س": "s", "ش": "x", "ص": "s", "ض": "z", "ط": "t",
    "ظ": "z", "ع": "", "غ": "g", "ف": "f", "ق": "k", "ک": "k", "گ": "g",
    "ل": "l", "م": "m", "ن": "n", "ں": "n", "و": "v", "ہ": "h", "ھ": "h",
    "ء": "", "ی": "", "ے": "", "ئ": "", "ؤ": "v", "ا": "", "آ": "", "أ": "",
    "ٰ": "", "ٗ": "", "ً": "", "ٌ": "", "ٍ": "", "َ": "", "ُ": "", "ِ": "",
    "ّ": "", "ْ": "",
}

_VOWELS = set("aeiou")


def is_urdu(s: str) -> bool:
    return any("؀" <= ch <= "ۿ" for ch in s)


def _dedup(s: str) -> str:
    """Collapse consecutive duplicates: gemination (Hussain vs Husain,
    double-m in Muhammad vs the Urdu shadda) carries no identity signal."""
    return "".join(ch for i, ch in enumerate(s) if i == 0 or s[i - 1] != ch)


def roman_skeleton(token: str) -> str:
    """Consonant skeleton of a Roman token: muhammad -> mhmd, khan -> kn."""
    t = re.sub(r"[^a-z]", "", token.lower())
    for digraph, repl in _DIGRAPHS:
        t = t.replace(digraph, repl)
    t = t.replace("q", "k").replace("w", "v").replace("y", "")
    out = [ch for ch in t if ch not in _VOWELS]
    return _dedup("".join(out))


def urdu_skeleton(token: str) -> str:
    """Consonant skeleton of an Urdu token via letter classes."""
    return _dedup("".join(_URDU.get(ch, "") for ch in token))


def skeleton(token: str) -> str:
    return urdu_skeleton(token) if is_urdu(token) else roman_skeleton(token)


def name_tokens(name: str) -> list[str]:
    """Cleaned lowercase tokens with honorifics stripped."""
    s = str(name).replace(".", " ").replace(",", " ").lower()
    toks = [t for t in re.split(r"\s+", s) if t]
    while toks and toks[0] in HONORIFICS:
        toks = toks[1:]
    return toks


def name_skeletons(name: str) -> list[str]:
    """Skeletons of all tokens, dropping empties (pure-vowel tokens)."""
    return [sk for t in name_tokens(name) if (sk := skeleton(t))]


CITY_ABBR = {
    "rwp": "rawalpindi", "isb": "islamabad", "lhr": "lahore", "lhe": "lahore",
    "khi": "karachi", "pesh": "peshawar", "fsd": "faisalabad",
    "mtn": "multan", "skt": "sialkot",
}

_ADDR_REPL = [
    (r"\bh\.?\s*no\.?\s*", "house "),
    (r"\bh\s*#\s*", "house "),
    (r"\bhouse\s*no\.?\s*", "house "),
    (r"\bplot\s+", "house "),
    (r"\bst\.?\s+", "street "),
    (r"\bgali\s+", "street "),
]


def normalize_address(addr: str) -> dict:
    """Parse a rendered address into comparable parts.

    Returns {house, street, area_tokens, city} with None for missing parts.
    """
    s = str(addr).lower().replace(",", " ")
    for pat, repl in _ADDR_REPL:
        s = re.sub(pat, repl, s)
    s = re.sub(r"\s+", " ", s).strip()

    house = street = None
    m = re.search(r"house\s+(\d+)", s)
    if m:
        house = int(m.group(1))
    m = re.search(r"street\s+(\d+)", s)
    if m:
        street = int(m.group(1))
    if house is None:
        m = re.match(r"^(\d+)-(\d+)\b", s)
        if m:
            house, street = int(m.group(1)), int(m.group(2))

    tokens = re.findall(r"[a-z][a-z\-/]+|\b[a-z]-?\d+(?:/\d+)?\b", s)
    city = None
    for t in reversed(tokens):
        t_clean = t.strip("-")
        if t_clean in CITY_ABBR:
            city = CITY_ABBR[t_clean]
            break
        if t_clean in ("rawalpindi", "islamabad", "lahore", "karachi",
                       "peshawar", "faisalabad", "multan", "sialkot"):
            city = t_clean
            break
    skip = {"house", "street", "no", city or ""} | set(CITY_ABBR)
    area = [t for t in tokens if t not in skip and not t.isdigit()]
    return {"house": house, "street": street, "area_tokens": area, "city": city}


def normalize_cnic(cnic: str) -> dict:
    """Returns {digits, masked_prefix, last} - any part may be None."""
    s = str(cnic).strip()
    if not s or s.lower() == "nan":
        return {"digits": None, "prefix": None, "last": None}
    digits = re.sub(r"[^0-9X*]", "", s.upper())
    if "X" in digits or "*" in digits:
        prefix = re.match(r"^(\d{5})", digits)
        last = re.search(r"(\d)$", digits)
        return {"digits": None,
                "prefix": prefix.group(1) if prefix else None,
                "last": last.group(1) if last else None}
    if len(digits) == 13:
        return {"digits": digits, "prefix": digits[:5], "last": digits[-1]}
    return {"digits": None, "prefix": None, "last": None}


def normalize_phone(phone: str) -> str | None:
    s = re.sub(r"\D", "", str(phone))
    if not s:
        return None
    if s.startswith("92"):
        s = "0" + s[2:]
    return s if len(s) == 11 else None
