"""Unit tests for the entity-resolution normalization primitives + edge cases."""
import entity_resolution as er


# ---- CNIC normalization --------------------------------------------------
def test_cnic_strips_formatting():
    assert er.normalize_cnic("12345-1234567-1") == "1234512345671"

def test_cnic_missing_and_garbage_become_empty():
    assert er.normalize_cnic("") == ""
    assert er.normalize_cnic(None) == ""
    assert er.normalize_cnic("N/A") == ""
    assert er.normalize_cnic("123") == ""            # wrong length -> unusable

def test_cnic_float_nan_safe():
    assert er.normalize_cnic(float("nan")) == ""


# ---- Name normalization (Urdu/English mixed text) ------------------------
def test_transliteration_variants_canonicalize_together():
    # the core challenge: different spellings of one name must collapse
    _, a = er.normalize_name("Muhammad Khan")
    _, b = er.normalize_name("Mohammad Khan")
    _, c = er.normalize_name("Mohammed Khan")
    assert a == b == c

def test_urdu_script_maps_to_latin_canonical():
    _, latin = er.normalize_name("Imran Mughal")
    _, urdu = er.normalize_name("عمران مغل")
    assert latin == urdu

def test_mixed_script_name():
    # a real record: one token Urdu, one Latin
    _, mixed = er.normalize_name("عمران Mughal")
    _, latin = er.normalize_name("Imran Mughal")
    assert mixed == latin

def test_name_similarity_high_for_variants():
    _, a = er.normalize_name("Ayesha Qureshi")
    _, b = er.normalize_name("Aisha Quraishi")
    assert er.name_similarity(a, b) > 0.8

def test_name_similarity_low_for_different_people():
    _, a = er.normalize_name("Imran Khan")
    _, b = er.normalize_name("Bilal Malik")
    assert er.name_similarity(a, b) < 0.3

def test_empty_name_edge_case():
    canon, toks = er.normalize_name("")
    assert canon == "" and toks == ()
    assert er.name_similarity(toks, ("imran",)) == 0.0


# ---- CNIC typo matching --------------------------------------------------
def test_typo_match_detects_single_digit_diff():
    assert er.cnic_typo_match("1234512345671", "1234512345672") is True

def test_typo_match_rejects_two_digit_diff():
    assert er.cnic_typo_match("1234512345671", "1234512345688") is False

def test_typo_match_rejects_empty():
    assert er.cnic_typo_match("", "1234512345671") is False
