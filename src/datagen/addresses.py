"""Pakistani address model with realistic rendering variation.

A canonical address is structured (house, street, area, city); registries
render it through different conventions: "House 12, Street 5", "H# 12 St 5",
"H.No. 12 Gali 5", city abbreviations like Rwp/Isb, uppercase, etc.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

CITIES: dict[str, dict] = {
    "Islamabad": {
        "abbr": ["Isb", "ISB"],
        "areas": ["F-7/2", "F-8/1", "G-9/4", "G-10/2", "G-11/3", "I-8/4",
                  "E-11/2", "DHA Phase 2", "Bahria Town Phase 4", "Bani Gala"],
    },
    "Rawalpindi": {
        "abbr": ["Rwp", "RWP"],
        "areas": ["Satellite Town", "Bahria Town Phase 7", "Chaklala Scheme 3",
                  "Saddar", "Westridge", "Peshawar Road", "Adiala Road",
                  "Gulraiz Housing Scheme"],
    },
    "Lahore": {
        "abbr": ["Lhr", "LHR"],
        "areas": ["DHA Phase 5", "Gulberg III", "Model Town", "Johar Town",
                  "Wapda Town", "Bahria Town Sector C", "Cantt",
                  "Allama Iqbal Town", "Garden Town", "Shadman"],
    },
    "Karachi": {
        "abbr": ["Khi", "KHI"],
        "areas": ["DHA Phase 6", "Clifton Block 2", "Gulshan-e-Iqbal Block 13",
                  "North Nazimabad Block H", "PECHS Block 6", "Bahadurabad",
                  "Malir Cantt"],
    },
    "Peshawar": {
        "abbr": ["Pesh"],
        "areas": ["Hayatabad Phase 3", "University Town", "Gulbahar",
                  "Warsak Road"],
    },
    "Faisalabad": {
        "abbr": ["Fsd"],
        "areas": ["Peoples Colony No 1", "Madina Town", "Susan Road",
                  "D Ground"],
    },
    "Multan": {
        "abbr": ["Mtn"],
        "areas": ["Cantt", "Gulgasht Colony", "Bosan Road",
                  "Shah Rukn-e-Alam Colony"],
    },
    "Sialkot": {
        "abbr": ["Skt"],
        "areas": ["Cantt", "Model Town", "Paris Road"],
    },
    "Hyderabad": {
        "abbr": ["Hyd"],
        "areas": ["Latifabad", "Qasimabad", "Saddar", "Citizen Colony"],
    },
    "Gujranwala": {
        "abbr": ["Gjw", "Grw"],
        "areas": ["Model Town", "Satellite Town", "Peoples Colony", "DC Colony"],
    },
    "Quetta": {
        "abbr": ["Qta"],
        "areas": ["Cantt", "Jinnah Town", "Satellite Town", "Samungli Road"],
    },
    "Bahawalpur": {
        "abbr": ["Bwp"],
        "areas": ["Model Town A", "Satellite Town", "Cantt"],
    },
}

CITY_NAMES = list(CITIES)
CITY_WEIGHTS = [14, 14, 22, 24, 8, 8, 6, 4, 7, 9, 4, 4]


@dataclass(frozen=True)
class Address:
    house: int
    street: int
    area: str
    city: str

    def canonical(self) -> str:
        return f"House {self.house}, Street {self.street}, {self.area}, {self.city}"


def random_address(rng: random.Random, city: str | None = None) -> Address:
    city = city or rng.choices(CITY_NAMES, weights=CITY_WEIGHTS)[0]
    return Address(
        house=rng.randint(1, 999),
        street=rng.randint(1, 60),
        area=rng.choice(CITIES[city]["areas"]),
        city=city,
    )


def render_address(addr: Address, rng: random.Random) -> str:
    city = addr.city
    abbrs = CITIES[city]["abbr"]
    city_out = rng.choice([city] * 3 + abbrs)
    style = rng.random()
    if style < 0.30:
        s = f"House {addr.house}, Street {addr.street}, {addr.area}, {city_out}"
    elif style < 0.50:
        s = f"H# {addr.house}, St {addr.street}, {addr.area}, {city_out}"
    elif style < 0.65:
        s = f"H.No. {addr.house}, Street {addr.street}, {addr.area}, {city_out}"
    elif style < 0.75:
        s = f"House No {addr.house} Gali {addr.street} {addr.area} {city_out}"
    elif style < 0.85:
        s = f"{addr.house}-{addr.street}, {addr.area}, {city_out}"
    else:
        s = f"Plot {addr.house}, {addr.area}, {city_out}"
    if rng.random() < 0.06:
        s = s.upper()
    return s
