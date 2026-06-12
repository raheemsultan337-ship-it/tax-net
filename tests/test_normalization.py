"""Unit tests for the entity-resolution cascade primitives + edge cases.

Targets the ported `matching` package: consonant-skeleton normalization (the
script-bridging trick), CNIC parsing (full / masked / garbage), the DOB band,
and the typo matcher.
"""
from matching import normalize as nz
from matching import features as ft
from matching.records import Record


def _rec(name="", *, father="", cnic="", dob=None, phone=None, city=None, addr=None):
    return Record(
        record_id="X", registry="test", raw_name=name,
        name_toks=nz.name_tokens(name) if not nz.is_urdu(name) else [],
        name_skels=nz.name_skeletons(name), urdu=nz.is_urdu(name),
        father_skels=nz.name_skeletons(father) if father else [],
        addr=nz.normalize_address(addr) if addr else None,
        cnic=nz.normalize_cnic(cnic), phone=nz.normalize_phone(phone) if phone else None,
        dob=dob, city=city,
    )


# ---- Consonant skeleton: the script-bridging core ------------------------
def test_urdu_and_roman_collapse_to_same_skeleton():
    assert nz.roman_skeleton("Muhammad") == nz.urdu_skeleton("محمد") == "mhmd"
    assert nz.roman_skeleton("Khan") == nz.urdu_skeleton("خان")

def test_roman_spelling_variants_collapse():
    sk = {nz.roman_skeleton(v) for v in ("Muhammad", "Mohammad", "Mohammed", "Muhammed")}
    assert sk == {"mhmd"}

def test_honorifics_are_stripped():
    assert nz.name_tokens("Dr. Imran Khan")[0] == "imran"
    assert nz.name_tokens("Haji Muhammad Akram")[0] == "muhammad"

def test_mixed_script_name_skeletons_agree():
    assert nz.name_skeletons("عمران Mughal") == nz.name_skeletons("Imran Mughal")


# ---- CNIC parsing (full / masked / garbage) ------------------------------
def test_cnic_full_parsed():
    c = nz.normalize_cnic("12345-1234567-1")
    assert c["digits"] == "1234512345671" and c["prefix"] == "12345" and c["last"] == "1"

def test_cnic_masked_keeps_prefix_and_last():
    c = nz.normalize_cnic("42101-XXXXXXX-7")
    assert c["digits"] is None and c["prefix"] == "42101" and c["last"] == "7"

def test_cnic_garbage_and_empty_are_none():
    for bad in ("", "N/A", "123", None, float("nan")):
        c = nz.normalize_cnic(bad)
        assert c["digits"] is None


# ---- Typo matcher (Damerau-1) --------------------------------------------
def test_damerau1_single_digit_diff():
    assert ft.damerau1("1234512345671", "1234512345681") is True

def test_damerau1_transposition():
    assert ft.damerau1("1234512345671", "1234512345617") is True

def test_damerau1_rejects_two_edits():
    assert ft.damerau1("1234512345671", "1234512345688") is False


# ---- Name similarity bands -----------------------------------------------
def test_name_band_high_for_variants():
    assert ft.band_name(_rec("Ayesha Qureshi"), _rec("Aisha Quraishi")) in ("name_exact", "name_close")

def test_name_band_diff_for_different_people():
    assert ft.band_name(_rec("Imran Khan"), _rec("Bilal Malik")) == "name_diff"


# ---- DOB band (the recall lever) -----------------------------------------
def test_dob_band_match_and_diff():
    assert ft.band_dob(_rec("A", dob="1980-05-12"), _rec("B", dob="1980-05-12")) == "dob_match"
    assert ft.band_dob(_rec("A", dob="1980-05-12"), _rec("B", dob="1991-01-02")) == "dob_diff"

def test_dob_band_unobservable_when_missing():
    assert ft.band_dob(_rec("A", dob=None), _rec("B", dob="1980-05-12")) is None
