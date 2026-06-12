"""
Stage 1 — Synthetic Pakistani civic data generator.

DESIGN PRINCIPLE (the wall):
  - We generate people whose *observable footprint* (vehicles, property, bills,
    travel) flows from their TRUE income via a behavioural model + noise.
  - Some people UNDER-REPORT income on their tax return. That under-reporting is
    generated INDEPENDENTLY of how the detector will later score people, so the
    ML solves a real inference problem (no circularity).
  - The detector reads ONLY data/observable/*.csv. It never sees true income or
    the is_evader label.
  - Ground truth (true income, is_evader, record->person linkage) is written to
    data/ground_truth/ and is opened ONLY by evaluate.py, AFTER detection.

We also inject realistic dirtiness so entity resolution is non-trivial:
  - name transliteration variants (Muhammad/Mohammad, Ayesha/Aisha, Urdu script)
  - missing / mistyped CNICs
  - address formatting differences
"""

import os
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBS_DIR = os.path.join(ROOT, "data", "observable")
GT_DIR = os.path.join(ROOT, "data", "ground_truth")
os.makedirs(OBS_DIR, exist_ok=True)
os.makedirs(GT_DIR, exist_ok=True)

N_PERSONS = 10000

# ---------------------------------------------------------------------------
# Name pools with transliteration variants (the Urdu/English mixed challenge)
# ---------------------------------------------------------------------------
FIRST_MALE = {
    "Muhammad": ["Muhammad", "Mohammad", "Mohammed", "Muhammed", "محمد"],
    "Ahmed": ["Ahmed", "Ahmad", "احمد"],
    "Ali": ["Ali", "علی"],
    "Hassan": ["Hassan", "Hasan", "حسن"],
    "Hussain": ["Hussain", "Husain", "Hussein", "حسین"],
    "Usman": ["Usman", "Osman", "Othman", "عثمان"],
    "Bilal": ["Bilal", "بلال"],
    "Hamza": ["Hamza", "Hamzah", "حمزہ"],
    "Imran": ["Imran", "عمران"],
    "Faisal": ["Faisal", "Faysal", "فیصل"],
}
FIRST_FEMALE = {
    "Ayesha": ["Ayesha", "Aisha", "Aysha", "Ayisha", "عائشہ"],
    "Fatima": ["Fatima", "Fatimah", "Fatema", "فاطمہ"],
    "Zainab": ["Zainab", "Zaynab", "زینب"],
    "Maryam": ["Maryam", "Mariam", "Marium", "مریم"],
    "Sana": ["Sana", "ثناء"],
    "Hira": ["Hira", "حرا"],
}
LAST = {
    "Khan": ["Khan", "خان"],
    "Malik": ["Malik", "Malick", "ملک"],
    "Sheikh": ["Sheikh", "Shaikh", "Shaykh", "شیخ"],
    "Chaudhry": ["Chaudhry", "Chaudhary", "Choudhary", "Ch.", "چوہدری"],
    "Butt": ["Butt", "Bhatti", "بٹ"],
    "Qureshi": ["Qureshi", "Quraishi", "قریشی"],
    "Syed": ["Syed", "Sayed", "Sayyid", "سید"],
    "Awan": ["Awan", "اعوان"],
    "Raja": ["Raja", "راجہ"],
    "Mughal": ["Mughal", "Moghul", "مغل"],
}

CITIES = {
    "Karachi": ["DHA", "Clifton", "Gulshan-e-Iqbal", "Malir", "North Nazimabad"],
    "Lahore": ["DHA", "Gulberg", "Model Town", "Johar Town", "Cantt"],
    "Islamabad": ["F-7", "F-8", "G-10", "E-11", "Bahria Town"],
    "Rawalpindi": ["Saddar", "Bahria Town", "Chaklala", "Satellite Town"],
    "Faisalabad": ["Madina Town", "Peoples Colony", "Jaranwala Road"],
}


def make_cnic():
    """Pakistani CNIC: 5 digits - 7 digits - 1 digit."""
    return f"{random.randint(10000,99999)}-{random.randint(1000000,9999999)}-{random.randint(0,9)}"


def corrupt_cnic(cnic, p_missing=0.12, p_typo=0.10):
    """Datasets are dirty: some CNICs missing, some have a digit typo."""
    r = random.random()
    if r < p_missing:
        return ""
    if r < p_missing + p_typo:
        digits = [c for c in cnic if c.isdigit()]
        idx = random.randrange(len(digits))
        digits[idx] = str(random.randint(0, 9))
        # re-insert into the dashed format
        d = iter(digits)
        return "".join(next(d) if ch.isdigit() else ch for ch in cnic)
    return cnic


def name_variant(canonical_first, canonical_last, gender):
    """Pick a transliteration variant — different datasets spell names differently."""
    fpool = (FIRST_MALE if gender == "M" else FIRST_FEMALE)[canonical_first]
    lpool = LAST[canonical_last]
    first = random.choice(fpool)
    last = random.choice(lpool)
    # sometimes a compound first name with "Muhammad" prefix for males
    if gender == "M" and canonical_first != "Muhammad" and random.random() < 0.25:
        pref = random.choice(FIRST_MALE["Muhammad"][:4])  # skip urdu prefix mix
        first = f"{pref} {first}"
    return f"{first} {last}"


def person_extra(p):
    """Noisy per-record copies of the disambiguating fields a real civic record
    carries: father's name (transliteration variants, sometimes missing) and
    date of birth (sometimes missing). These let entity resolution tell apart
    two same-named people who lack/share a CNIC."""
    # father shares the family surname; first name is a male transliteration variant
    if random.random() < 0.15:
        father = ""                                   # missing in this dataset
    else:
        father = name_variant(p.father_first, p.canonical_last, "M")
    dob = "" if random.random() < 0.10 else p.dob     # occasionally missing
    return dob, father


# ---------------------------------------------------------------------------
# Behavioural model: footprint flows from TRUE income; under-reporting is
# generated independently of the detection logic.
# ---------------------------------------------------------------------------
def _new_dob():
    return f"{random.randint(1955, 2003)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"


def build_persons():
    persons = []
    for pid in range(N_PERSONS):
        gender = random.choice(["M", "M", "M", "F"])  # skew matches filer demographics
        cfirst = random.choice(list((FIRST_MALE if gender == "M" else FIRST_FEMALE).keys()))
        clast = random.choice(list(LAST.keys()))
        city = random.choice(list(CITIES.keys()))
        area = random.choice(CITIES[city])
        cnic = make_cnic()
        # disambiguating identity fields (real civic records carry these)
        father_first = random.choice(list(FIRST_MALE.keys()))
        dob = _new_dob()
        # fine-grained household address (a real street address). Proxies will
        # SHARE their principal's household — that shared node is the graph link.
        household = f"H-{pid:05d}, {area}, {city}"

        # TRUE annual income (PKR), lognormal -> long right tail of wealthy people
        true_income = float(np.random.lognormal(mean=14.0, sigma=1.05))  # wide, fat-tailed

        # Compliance is a SPECTRUM, drawn independently of the detection logic
        # (no circularity). Most people declare honestly; a band under-reports to
        # varying degrees (still FILING, non-zero declared); a tail evades hard.
        # Serious evasion concentrates among the WEALTHY — they have the most to
        # hide AND the visible footprint that gives them away. We scale up their
        # true income (bigger footprint) but most of them still FILE a tiny return,
        # so they show up as flaggable NON-ZERO-declared cases, not just non-filers.
        r = random.random()
        if r < 0.52:                                    # honest
            report_ratio = float(np.random.uniform(0.85, 1.00))
            non_filer = random.random() < 0.02
        elif r < 0.75:                                  # mild→moderate under-reporter (files; mostly compliant)
            report_ratio = float(np.random.uniform(0.45, 0.80))
            non_filer = False
        else:                                           # serious evader (wealthy; almost all FILE a tiny return)
            true_income *= float(np.random.uniform(2.0, 4.5))   # scale up the visible footprint
            report_ratio = float(np.random.uniform(0.04, 0.22))  # declares only a small slice of it
            non_filer = random.random() < 0.10
        declared_income = 0.0 if non_filer else true_income * report_ratio

        # Ground-truth (read ONLY by evaluate.py): materially under-reported —
        # declared less than half of true income. The detector never sees this.
        is_evader = declared_income < 0.5 * true_income

        persons.append({
            "person_id": pid,
            "gender": gender,
            "canonical_first": cfirst,
            "canonical_last": clast,
            "father_first": father_first,
            "dob": dob,
            "cnic": cnic,
            "city": city,
            "area": area,
            "household": household,
            "true_income": round(true_income),
            # asset_income drives the OBSERVABLE footprint (vehicles/property/
            # utilities). For most people it equals true_income; for principals
            # it shrinks (assets moved to proxies); for proxies it is the hidden
            # chunk they hold.
            "asset_income": round(true_income),
            "report_ratio": round(report_ratio, 3),
            "declared_income": round(declared_income),
            "non_filer": non_filer,
            "is_evader": is_evader,
            "role": "evader" if is_evader else "normal",
            "principal_id": -1,        # which person this proxy fronts for (-1 = none)
            "uses_proxy": False,
        })

    persons = _inject_proxies(persons)
    return pd.DataFrame(persons)


def _inject_proxies(persons):
    """Sophisticated evaders ('principals') hide assets under proxies/frontmen.

    A principal keeps a small VISIBLE footprint (and files a clean-looking return
    matching it), while the bulk of their wealth is registered to 1-2 proxies who
    share the principal's household and carry the principal's name as their
    father's name. The principal therefore looks compliant on their own books —
    only the graph (shared household -> non-filing asset-holders) reveals them."""
    next_pid = len(persons)
    base = list(persons)               # snapshot before appending proxies
    for person in base:
        if not (person["is_evader"] and person["true_income"] > 6_000_000):
            continue
        if random.random() >= 0.35:    # a minority of the wealthiest evaders use proxies
            continue

        hidden_frac = random.uniform(0.55, 0.85)
        visible = person["true_income"] * (1 - hidden_frac)
        # principal now looks compliant: footprint ~ visible, files matching return
        person["asset_income"] = round(visible)
        person["declared_income"] = round(visible * random.uniform(0.85, 1.05))
        person["non_filer"] = False
        person["role"] = "principal"
        person["uses_proxy"] = True

        n_proxies = random.randint(1, 2)
        per_proxy_hidden = person["true_income"] * hidden_frac / n_proxies
        for _ in range(n_proxies):
            pgender = random.choice(["M", "M", "F"])
            pfirst = random.choice(list((FIRST_MALE if pgender == "M"
                                         else FIRST_FEMALE).keys()))
            persons.append({
                "person_id": next_pid,
                "gender": pgender,
                "canonical_first": pfirst,
                "canonical_last": person["canonical_last"],     # same family name as principal
                "father_first": random.choice(list(FIRST_MALE.keys())),
                "dob": _new_dob(),
                "cnic": make_cnic(),
                "city": person["city"],
                "area": person["area"],
                "household": person["household"],                # SHARED address (the link)
                "true_income": round(per_proxy_hidden * random.uniform(0.05, 0.15)),
                "asset_income": round(per_proxy_hidden),         # holds the hidden assets
                "report_ratio": 0.0,
                "declared_income": 0.0,
                "non_filer": True,                               # frontmen don't file
                "is_evader": False,                              # not the beneficial owner
                "role": "proxy",
                "principal_id": person["person_id"],
                "uses_proxy": False,
            })
            next_pid += 1
    return persons


# ---------------------------------------------------------------------------
# Observable record generators. Each emits noisy, separately-keyed records.
# We track the true person_id ONLY in a linkage file (for ER evaluation).
# ---------------------------------------------------------------------------
def gen_vehicles(persons):
    rows, link = [], []
    rid = 0
    for p in persons.itertuples():
        # number of vehicles scales (loosely) with the OBSERVABLE asset income
        n = np.random.poisson(max(0.2, p.asset_income / 4_000_000))
        n = min(n, 4)
        for _ in range(n):
            # engine cc correlates with income (moderate noise)
            base = 800 + (p.asset_income / 1_000_000) * 320
            cc = int(np.clip(np.random.normal(base, 150), 660, 4500))
            cc = round(cc / 100) * 100
            reg_value = int(cc * np.random.uniform(3200, 4200))
            dob, father = person_extra(p)
            rows.append({
                "record_id": f"VEH{rid:06d}",
                "owner_name": name_variant(p.canonical_first, p.canonical_last, p.gender),
                "owner_cnic": corrupt_cnic(p.cnic),
                "father_name": father,
                "dob": dob,
                "engine_cc": cc,
                "reg_value": reg_value,
                "city": p.city,
            })
            link.append({"record_id": f"VEH{rid:06d}", "person_id": p.person_id})
            rid += 1
    return pd.DataFrame(rows), pd.DataFrame(link)


def gen_real_estate(persons):
    rows, link = [], []
    rid = 0
    for p in persons.itertuples():
        n = np.random.poisson(max(0.15, p.asset_income / 6_000_000))
        n = min(n, 3)
        for _ in range(n):
            # property value ~ 3.5x annual income (tight lognormal noise)
            base_val = p.asset_income * float(np.random.lognormal(np.log(3.5), 0.22))
            val = int(np.clip(base_val, 2_000_000, 500_000_000))
            dob, father = person_extra(p)
            rows.append({
                "record_id": f"PROP{rid:06d}",
                "owner_name": name_variant(p.canonical_first, p.canonical_last, p.gender),
                "owner_cnic": corrupt_cnic(p.cnic),
                "father_name": father,
                "dob": dob,
                "property_value": val,
                "household": p.household,
                "area": p.area,
                "city": p.city,
            })
            link.append({"record_id": f"PROP{rid:06d}", "person_id": p.person_id})
            rid += 1
    return pd.DataFrame(rows), pd.DataFrame(link)


def gen_utilities(persons):
    rows, link = [], []
    rid = 0
    for p in persons.itertuples():
        # nearly everyone has one electricity account; bill scales with income
        base_bill = 3000 + (p.asset_income / 1_000_000) * 9000
        bill = int(np.clip(np.random.normal(base_bill, base_bill * 0.12), 1500, 1_500_000))
        dob, father = person_extra(p)
        rows.append({
            "account_id": f"UTL{rid:06d}",
            "consumer_name": name_variant(p.canonical_first, p.canonical_last, p.gender),
            "consumer_cnic": corrupt_cnic(p.cnic, p_missing=0.20),  # utilities dirtier
            "father_name": father,
            "dob": dob,
            "address": p.household,
            "avg_monthly_bill": bill,
        })
        link.append({"record_id": f"UTL{rid:06d}", "person_id": p.person_id})
        rid += 1
    return pd.DataFrame(rows), pd.DataFrame(link)


def gen_travel(persons):
    rows, link = [], []
    rid = 0
    dests = ["Dubai", "London", "Jeddah", "Istanbul", "Bangkok", "New York", "Toronto"]
    for p in persons.itertuples():
        # travel scales with observable asset income (a careful principal also
        # curbs visible luxury), so principals stay clean on their own books.
        rate = max(0.05, p.asset_income / 3_000_000)
        trips = np.random.poisson(rate)
        if trips == 0:
            continue
        trips = min(trips, 12)
        dob, father = person_extra(p)
        rows.append({
            "record_id": f"TRV{rid:06d}",
            "passenger_name": name_variant(p.canonical_first, p.canonical_last, p.gender),
            "passport_cnic": corrupt_cnic(p.cnic),
            "father_name": father,
            "dob": dob,
            "trips_last_year": int(trips),
            "destinations": ", ".join(random.sample(dests, min(len(dests), max(1, trips // 2 + 1)))),
        })
        link.append({"record_id": f"TRV{rid:06d}", "person_id": p.person_id})
        rid += 1
    return pd.DataFrame(rows), pd.DataFrame(link)


def gen_tax_returns(persons):
    """Tax return reflects DECLARED income (post under-reporting)."""
    rows, link = [], []
    rid = 0
    for p in persons.itertuples():
        # Non-filers (latent flag) file no return at all.
        if p.non_filer:
            continue
        declared = p.declared_income
        # simplified progressive tax
        tax = int(max(0, (declared - 600_000) * 0.15))
        dob, father = person_extra(p)
        rows.append({
            "return_id": f"TAX{rid:06d}",
            "filer_name": name_variant(p.canonical_first, p.canonical_last, p.gender),
            "filer_cnic": corrupt_cnic(p.cnic, p_missing=0.05),  # tax records cleaner
            "father_name": father,
            "dob": dob,
            "declared_income": int(declared),
            "tax_paid": tax,
        })
        link.append({"record_id": f"TAX{rid:06d}", "person_id": p.person_id})
        rid += 1
    return pd.DataFrame(rows), pd.DataFrame(link)


def main():
    persons = build_persons()

    datasets = {
        "vehicles": gen_vehicles(persons),
        "real_estate": gen_real_estate(persons),
        "utilities": gen_utilities(persons),
        "travel": gen_travel(persons),
        "tax_returns": gen_tax_returns(persons),
    }

    all_links = []
    for name, (df, link) in datasets.items():
        df.to_csv(os.path.join(OBS_DIR, f"{name}.csv"), index=False)
        link["source"] = name
        all_links.append(link)
        print(f"  observable/{name}.csv  ->  {len(df):>5} records")

    # Ground truth (NEVER read by the detector)
    persons.to_csv(os.path.join(GT_DIR, "persons.csv"), index=False)
    pd.concat(all_links, ignore_index=True).to_csv(
        os.path.join(GT_DIR, "record_linkage.csv"), index=False
    )

    n_ev = int(persons["is_evader"].sum())
    print(f"\n  ground_truth/persons.csv         ->  {len(persons)} persons")
    print(f"  ground_truth/record_linkage.csv  ->  {sum(len(l) for l in all_links)} record->person links")
    print(f"  planted evaders (is_evader)      ->  {n_ev} ({n_ev/len(persons)*100:.1f}%)")
    print("\nDone. Detector reads ONLY data/observable/*.csv")


if __name__ == "__main__":
    main()
