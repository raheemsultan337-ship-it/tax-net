"""Population model: persons with income profiles, compliance segments,
and income-driven asset plans (vehicles, property, electricity, travel).

Each person carries a hidden person_id. The asset plan drives what shows up
in each registry; the compliance segment drives the gap between actual and
declared income. Hand-seeded edge personas guarantee the demo story and the
hard ER test cases exist regardless of the random draw.

Adapted from the donor generator: adds a full ``dob`` (the ER recall lever) and
tax-net's latent labels (``role`` principal/proxy, ``principal_id``). The
proxy/benami model is reworked so the principal keeps a SMALL visible footprint
INCLUDING a utility at the shared address (so the knowledge graph can link them),
while the bulk of the wealth sits with a non-filing proxy — only the shared
address reveals the hidden assets. These labels live behind the wall.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .tax import annual_tax
from . import names
from .addresses import Address, random_address
from .noise import make_cnic, make_phone

SEGMENTS = {
    "ordinary_compliant": 0.46,
    "ordinary_nonfiler": 0.17,
    "wealthy_compliant": 0.08,
    "wealthy_nonfiler": 0.06,
    "under_reporter": 0.18,   # more under-reporting FILERS → fills the scatter's middle
    "proxy_pair": 0.05,
}

# An evader is someone materially under-reporting a materially-taxable income.
# The actual-income floor keeps legal below-threshold non-filers OUT of the
# target set (so the detector isn't graded on flagging the poor).
EVADER_MIN_INCOME = 2_500_000

VEHICLES_BUDGET = [
    (700_000, [("Honda CD-70", 70), ("Honda CG-125", 125), ("Yamaha YBR 125", 124)]),
    (1_500_000, [("Suzuki Mehran", 796), ("Suzuki Alto", 660), ("Suzuki Wagon R", 998),
                 ("Suzuki Cultus", 998)]),
    (3_500_000, [("Toyota Corolla GLi", 1299), ("Honda City", 1497),
                 ("Toyota Yaris", 1329), ("MG HS", 1490)]),
    (8_000_000, [("Honda Civic", 1799), ("Toyota Corolla Altis", 1797),
                 ("Kia Sportage", 1999), ("Hyundai Tucson", 1999)]),
    (float("inf"), [("Toyota Fortuner", 2694), ("Toyota Land Cruiser Prado", 2982),
                    ("Toyota Hilux Revo", 2755), ("Audi A4", 1984),
                    ("Mercedes C200", 1991), ("BMW X5", 2998)]),
]

DESTINATIONS = ["Dubai", "Jeddah", "London", "Istanbul", "Bangkok",
                "Kuala Lumpur", "Doha", "Toronto", "New York", "Baku"]


@dataclass
class Person:
    person_id: str
    gender: str
    name_tokens: list[str]
    father_tokens: list[str]
    cnic: str
    birth_year: int
    address: Address
    phones: list[str]
    segment: str
    actual_income: int          # PKR per year, lifestyle-supporting income
    declared_income: int        # what they tell FBR
    filer_status: str           # filer | late_filer | nil_filer | absent
    vehicles: list[tuple[str, int, int]] = field(default_factory=list)   # make, cc, year
    properties: list[tuple[int, int]] = field(default_factory=list)      # dc_value, marla
    avg_monthly_bill: int = 0
    trips_abroad: int = 0
    days_abroad: int = 0
    destinations: list[str] = field(default_factory=list)
    n_sims: int = 1
    household_head: str | None = None   # person_id paying the electricity bill
    note: str = ""
    dob: str = ""                       # YYYY-MM-DD (the ER recall lever)
    role: str = "normal"                # normal | principal | proxy
    principal_id: str = ""              # which principal this proxy fronts for


def _income_for(segment: str, rng: random.Random) -> int:
    if segment.startswith("ordinary"):
        return int(rng.lognormvariate(13.6, 0.62))           # wider spread (~0.3M - 4M)
    if segment == "under_reporter":
        return int(rng.uniform(3_000_000, 45_000_000))       # wider income range
    return int(rng.uniform(8_000_000, 90_000_000))           # wealthy_*, taller tail


def _declared_for(segment: str, actual: int, rng: random.Random) -> tuple[int, str]:
    if segment in ("ordinary_compliant", "wealthy_compliant"):
        declared = int(actual * rng.uniform(0.88, 1.0))
        return declared, "filer"
    if segment == "ordinary_nonfiler":
        # Below or near the taxable threshold - non-filing is legal here.
        return 0, "absent"
    if segment == "under_reporter":
        # a continuum of under-reporting (8%–55% declared) → a spread of points
        # across the gap region of the scatter, not a single tight cluster
        return int(actual * rng.uniform(0.08, 0.55)), "filer"
    # wealthy_nonfiler: ghosts (absent) or nil-filers
    if rng.random() < 0.6:
        return 0, "absent"
    return int(rng.uniform(0, 800_000)), "nil_filer"


def _plan_assets(p: Person, rng: random.Random) -> None:
    inc = p.actual_income
    year = lambda: rng.randint(2014, 2025)
    n_veh = 0
    if inc > 8_000_000:
        n_veh = rng.randint(1, 3)
    elif inc > 2_500_000:
        n_veh = rng.choices([0, 1, 2], weights=[2, 7, 1])[0]
    elif inc > 900_000:
        n_veh = rng.choices([0, 1], weights=[4, 6])[0]
    else:
        n_veh = rng.choices([0, 1], weights=[5, 5])[0]
    for _ in range(n_veh):
        for budget, pool in VEHICLES_BUDGET:
            if inc <= budget:
                make, cc = rng.choice(pool)
                p.vehicles.append((make, cc, year()))
                break

    if inc > 8_000_000:
        for _ in range(rng.randint(1, 4)):
            p.properties.append((int(rng.uniform(8e6, 1.2e8)), rng.choice([5, 7, 10, 20, 40])))
    elif inc > 3_000_000 and rng.random() < 0.4:
        p.properties.append((int(rng.uniform(3e6, 1.5e7)), rng.choice([3, 5, 7])))

    rate = rng.uniform(0.015, 0.05) if inc < 8_000_000 else rng.uniform(0.02, 0.06)
    p.avg_monthly_bill = max(2_500, int(inc * rate / 12))

    if inc > 8_000_000:
        p.trips_abroad = rng.randint(1, 8)
    elif inc > 2_500_000 and rng.random() < 0.3:
        p.trips_abroad = rng.randint(1, 2)
    if p.trips_abroad:
        p.days_abroad = p.trips_abroad * rng.randint(5, 18)
        p.destinations = rng.sample(DESTINATIONS, min(p.trips_abroad, 4))

    p.n_sims = rng.choices([1, 2, 3], weights=[6, 3, 1])[0]


def _birthdate(birth_year: int, rng: random.Random) -> str:
    return f"{birth_year}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


def _new_person(pid: str, segment: str, rng: random.Random,
                gender: str | None = None) -> Person:
    gender = gender or ("M" if rng.random() < 0.7 else "F")
    tokens = names.pick_name_tokens(rng, gender)
    family = tokens[-1] if tokens[-1] not in ("Bibi", "Begum") else None
    father = names.pick_name_tokens(rng, "M", family=family)
    addr = random_address(rng)
    actual = _income_for(segment, rng)
    declared, status = _declared_for(segment, actual, rng)
    by = rng.randint(1950, 2000)
    p = Person(
        person_id=pid, gender=gender, name_tokens=tokens, father_tokens=father,
        cnic=make_cnic(rng, addr.city), birth_year=by,
        address=addr, phones=[make_phone(rng)], segment=segment,
        actual_income=actual, declared_income=declared, filer_status=status,
        dob=_birthdate(by, rng),
    )
    _plan_assets(p, rng)
    return p


def _proxy_pair(pid_a: str, pid_b: str, rng: random.Random) -> list[Person]:
    """Wealthy under-reporter ('principal') parks the bulk of his wealth with a
    no-income relative ('proxy').

    The principal keeps a SMALL visible footprint — crucially a utility account at
    the shared household — and files a clean-looking return, so his own books look
    compliant. The proxy (a non-filer) holds the big property + extra vehicles at
    the SAME address. Only the shared-address link to that non-filing asset-holder
    exposes the principal: own-only scoring misses him, the graph catches him.
    """
    owner = _new_person(pid_a, "under_reporter", rng, gender="M")
    owner.note = "proxy_owner_real"
    owner.role = "principal"
    owner.principal_id = ""

    proxy = _new_person(pid_b, "ordinary_nonfiler", rng, gender="F")
    proxy.note = "proxy_holder"
    proxy.role = "proxy"
    proxy.principal_id = owner.person_id
    proxy.address = owner.address                       # SHARED address (the link)
    proxy.cnic = make_cnic(rng, owner.address.city)
    proxy.name_tokens = [rng.choice(names.FEMALE_FIRST), rng.choice(["Bibi", "Begum"])]
    proxy.father_tokens = owner.father_tokens           # same family father
    proxy.phones = [owner.phones[0]]                    # shared phone (a second link)
    proxy.actual_income = 0
    proxy.declared_income = 0
    proxy.filer_status = "absent"

    # Guarantee a sizeable hidden property, then move the bulk of wealth to the proxy.
    if not owner.properties:
        owner.properties = [(int(rng.uniform(2.5e7, 1.2e8)), rng.choice([10, 20, 40]))]
    proxy.properties = owner.properties
    proxy.vehicles = owner.vehicles[1:]
    proxy.avg_monthly_bill = max(owner.avg_monthly_bill, 120_000)
    proxy.household_head = None                         # proxy holds property (address-bearing)
    proxy.trips_abroad, proxy.days_abroad, proxy.destinations = 0, 0, []

    # Principal: keep at most one modest vehicle + a small utility at the shared
    # address; he still files his under-reporter return.
    owner.properties = []
    owner.vehicles = owner.vehicles[:1]
    owner.avg_monthly_bill = max(8_000, int(owner.avg_monthly_bill * 0.18))
    owner.household_head = None                         # owner GETS a utility record
    owner.trips_abroad, owner.days_abroad, owner.destinations = 0, 0, []
    return [owner, proxy]


def seed_edge_personas(rng: random.Random) -> list[Person]:
    """Hand-built personas: the demo star and the matcher's hardest cases."""
    out: list[Person] = []
    rwp = Address(241, 12, "Bahria Town Phase 7", "Rawalpindi")

    star = Person(
        person_id="P000001", gender="M",
        name_tokens=["Ahmed", "Raza", "Khan"],
        father_tokens=["Abdul", "Khan"],
        cnic="37405-6783921-7", birth_year=1975, address=rwp,
        phones=["03215554417", "03335554417"], segment="wealthy_nonfiler",
        actual_income=85_000_000, declared_income=600_000, filer_status="nil_filer",
        vehicles=[("Toyota Land Cruiser Prado", 2982, 2022), ("Honda Civic", 1799, 2019)],
        properties=[(25_000_000, 10), (48_000_000, 20), (90_000_000, 40)],
        avg_monthly_bill=280_000, trips_abroad=6, days_abroad=70,
        destinations=["Dubai", "London", "Istanbul", "Doha"], n_sims=2,
        note="demo_star",
    )
    out.append(star)

    lhr = Address(77, 8, "Allama Iqbal Town", "Lahore")
    father = Person(
        person_id="P000002", gender="M",
        name_tokens=["Muhammad", "Akram", "Bhatti"], father_tokens=["Fazal", "Bhatti"],
        cnic="35202-1948374-1", birth_year=1955, address=lhr,
        phones=["03004412987"], segment="ordinary_compliant",
        actual_income=1_400_000, declared_income=1_300_000, filer_status="filer",
        vehicles=[("Toyota Corolla GLi", 1299, 2016)], avg_monthly_bill=18_000,
        note="father_son_father",
    )
    son = Person(
        person_id="P000003", gender="M",
        name_tokens=["Muhammad", "Akram", "Bhatti"],
        father_tokens=["Muhammad", "Akram", "Bhatti"],
        cnic="35202-8812245-9", birth_year=1988, address=lhr,
        phones=["03331107865"], segment="ordinary_compliant",
        actual_income=950_000, declared_income=900_000, filer_status="filer",
        vehicles=[("Honda CG-125", 125, 2021)], avg_monthly_bill=0,
        household_head="P000002", note="father_son_son",
    )
    out += [father, son]

    khi = Address(33, 4, "Gulshan-e-Iqbal Block 13", "Karachi")
    twin_a = Person(
        person_id="P000004", gender="M",
        name_tokens=["Hassan", "Raza", "Naqvi"], father_tokens=["Sajjad", "Naqvi"],
        cnic="42101-5567120-3", birth_year=1990, address=khi,
        phones=["03452219840"], segment="ordinary_compliant",
        actual_income=2_200_000, declared_income=2_000_000, filer_status="filer",
        vehicles=[("Suzuki Cultus", 998, 2020)], avg_monthly_bill=22_000,
        note="twin_a",
    )
    twin_b = Person(
        person_id="P000005", gender="M",
        name_tokens=["Hussain", "Raza", "Naqvi"], father_tokens=["Sajjad", "Naqvi"],
        cnic="42101-5567121-5", birth_year=1990, address=khi,
        phones=["03452219841"], segment="ordinary_compliant",
        actual_income=2_400_000, declared_income=2_200_000, filer_status="filer",
        vehicles=[("Honda City", 1497, 2021)], avg_monthly_bill=0,
        household_head="P000004", note="twin_b",
    )
    out += [twin_a, twin_b]

    isb = Address(18, 3, "F-8/1", "Islamabad")
    clean = Person(
        person_id="P000006", gender="F",
        name_tokens=["Samina", "Qureshi"], father_tokens=["Bashir", "Qureshi"],
        cnic="61101-2244913-8", birth_year=1968, address=isb,
        phones=["03008881234"], segment="wealthy_compliant",
        actual_income=18_000_000, declared_income=17_500_000, filer_status="filer",
        vehicles=[("Toyota Fortuner", 2694, 2023)],
        properties=[(60_000_000, 20)], avg_monthly_bill=90_000,
        trips_abroad=2, days_abroad=20, destinations=["London", "Jeddah"],
        note="wealthy_compliant_control",
    )
    out.append(clean)

    pesh = Address(9, 2, "Hayatabad Phase 3", "Peshawar")
    overseas = Person(
        person_id="P000007", gender="M",
        name_tokens=["Imran", "Gilani"], father_tokens=["Sadiq", "Gilani"],
        cnic="17301-7733402-2", birth_year=1980, address=pesh,
        phones=["03459902211"], segment="wealthy_nonfiler",
        actual_income=20_000_000, declared_income=0, filer_status="absent",
        properties=[(35_000_000, 10), (28_000_000, 7)],
        avg_monthly_bill=9_000, trips_abroad=3, days_abroad=210,
        destinations=["Toronto", "Dubai"], note="overseas_resident",
    )
    out.append(overseas)

    # Proxy/benami seed: principal keeps a small utility at the shared address and
    # files; the wife (proxy, non-filer) holds the big property + vehicle.
    lhr2 = Address(402, 21, "DHA Phase 5", "Lahore")
    husband = Person(
        person_id="P000008", gender="M",
        name_tokens=["Tariq", "Awan"], father_tokens=["Liaquat", "Awan"],
        cnic="35202-6190773-4", birth_year=1970, address=lhr2,
        phones=["03214090911"], segment="under_reporter",
        actual_income=40_000_000, declared_income=1_200_000, filer_status="filer",
        avg_monthly_bill=15_000, note="proxy_owner_real_seed",
        role="principal",
    )
    wife = Person(
        person_id="P000009", gender="F",
        name_tokens=["Bushra", "Begum"], father_tokens=["Liaquat", "Awan"],
        cnic="35202-9981230-6", birth_year=1974, address=lhr2,
        phones=["03214090911"], segment="ordinary_nonfiler",
        actual_income=0, declared_income=0, filer_status="absent",
        vehicles=[("Toyota Land Cruiser Prado", 2982, 2021)],
        properties=[(70_000_000, 20)], avg_monthly_bill=220_000,
        note="proxy_holder_seed", role="proxy", principal_id="P000008",
    )
    out += [husband, wife]
    return out


def generate_population(n_persons: int, rng: random.Random) -> list[Person]:
    persons = seed_edge_personas(rng)
    next_id = len(persons) + 1
    segs = list(SEGMENTS)
    weights = list(SEGMENTS.values())
    while len(persons) < n_persons:
        seg = rng.choices(segs, weights=weights)[0]
        if seg == "proxy_pair" and len(persons) + 2 <= n_persons:
            pair = _proxy_pair(f"P{next_id:06d}", f"P{next_id + 1:06d}", rng)
            persons += pair
            next_id += 2
        else:
            if seg == "proxy_pair":
                seg = "under_reporter"
            persons.append(_new_person(f"P{next_id:06d}", seg, rng))
            next_id += 1

    # Fill any missing DOB (seed personas) deterministically.
    for p in persons:
        if not p.dob:
            p.dob = _birthdate(p.birth_year, rng)
    return persons


def is_evader(p: Person) -> bool:
    """Latent ground-truth label (behind the wall): materially under-reporting a
    materially-taxable income, or a proxy-using principal."""
    if p.role == "principal":
        return True
    if p.role == "proxy":
        return False
    return p.declared_income < 0.5 * p.actual_income and p.actual_income >= EVADER_MIN_INCOME
