"""Pakistani name pools, spelling variants, and Urdu script renderings.

A person's canonical name is a list of tokens (e.g. ["Muhammad", "Asif", "Khan"]).
Registries render those tokens through different styles: variant spellings,
abbreviations, honorifics, Urdu script, ALL CAPS, typos. The variation is the
whole point — it is what the entity resolution pipeline must see through.
"""

from __future__ import annotations

import random

MALE_FIRST = [
    "Ahmed", "Ali", "Hassan", "Hussain", "Usman", "Bilal", "Imran", "Asif",
    "Tariq", "Khalid", "Faisal", "Kamran", "Shahid", "Zafar", "Rashid",
    "Naveed", "Salman", "Adnan", "Farhan", "Junaid", "Kashif", "Nadeem",
    "Saeed", "Waqar", "Yasir", "Zubair", "Arif", "Javed", "Iqbal", "Akram",
    "Aslam", "Bashir", "Ejaz", "Fahad", "Hamid", "Irfan", "Jamil", "Karim",
    "Mansoor", "Nasir", "Qasim", "Raza", "Sajid", "Talha", "Umar", "Wasim",
    "Zahid", "Rafiq", "Shoaib", "Tanveer", "Mazhar", "Nawaz", "Sohail",
    "Liaquat", "Pervez", "Younis", "Mehmood", "Shafiq", "Anwar", "Riaz",
]

FEMALE_FIRST = [
    "Ayesha", "Fatima", "Zainab", "Maryam", "Khadija", "Sana", "Hira",
    "Saba", "Nida", "Rabia", "Sadia", "Farah", "Uzma", "Shazia", "Nazia",
    "Samina", "Bushra", "Asma", "Lubna", "Tahira", "Amna", "Iqra", "Kiran",
    "Mehwish", "Nimra", "Rukhsana", "Shabana", "Yasmin", "Zubaida", "Naila",
    "Shaista", "Robina", "Nasreen", "Shagufta", "Abida",
]

FAMILY = [
    "Khan", "Malik", "Butt", "Sheikh", "Chaudhry", "Qureshi", "Siddiqui",
    "Awan", "Abbasi", "Raja", "Mirza", "Baig", "Ansari", "Hashmi", "Gilani",
    "Bhatti", "Dar", "Lodhi", "Niazi", "Mughal", "Shah", "Paracha",
    "Khattak", "Yousafzai", "Afridi", "Satti", "Kayani", "Janjua", "Naqvi",
    "Rizvi", "Zaidi", "Kazmi", "Hamdani", "Abbas", "Akhtar",
]

# Alternative Roman spellings seen across real registries.
VARIANTS: dict[str, list[str]] = {
    "Muhammad": ["Mohammad", "Mohd", "Muhammed", "Mohammed", "M."],
    "Ahmed": ["Ahmad"],
    "Hassan": ["Hasan"],
    "Hussain": ["Husain", "Hussein", "Hussian"],
    "Usman": ["Osman", "Uthman"],
    "Faisal": ["Faysal", "Feisal"],
    "Rashid": ["Rasheed"],
    "Saeed": ["Said", "Saied"],
    "Khalid": ["Khaled"],
    "Umar": ["Omer", "Omar"],
    "Karim": ["Kareem"],
    "Rafiq": ["Rafique"],
    "Tanveer": ["Tanvir"],
    "Shoaib": ["Shuaib", "Shoib"],
    "Javed": ["Javaid", "Javid"],
    "Mehmood": ["Mahmood", "Mehmud"],
    "Naveed": ["Navid"],
    "Zubair": ["Zubayr", "Zobair"],
    "Wasim": ["Waseem"],
    "Nadeem": ["Nadim"],
    "Shafiq": ["Shafique"],
    "Riaz": ["Riyaz"],
    "Sohail": ["Suhail", "Sohel"],
    "Aslam": ["Islam"],
    "Qasim": ["Qasem", "Kasim"],
    "Sheikh": ["Shaikh", "Shiekh", "Sh."],
    "Chaudhry": ["Chaudhary", "Choudhry", "Chohan", "Ch."],
    "Qureshi": ["Quraishi", "Qureishi"],
    "Siddiqui": ["Siddiqi", "Sidiqui"],
    "Gilani": ["Gillani", "Geelani"],
    "Baig": ["Beg", "Begg"],
    "Yousafzai": ["Yusufzai"],
    "Ayesha": ["Aisha", "Aysha"],
    "Fatima": ["Fatimah", "Fathima"],
    "Zainab": ["Zaynab", "Zenab"],
    "Maryam": ["Mariam", "Marium"],
    "Sadia": ["Saadia", "Sadiya"],
    "Yasmin": ["Yasmeen"],
    "Khadija": ["Khadijah", "Khadeeja"],
    "Nasreen": ["Nasrin"],
    "Samina": ["Sameena"],
    "Tahira": ["Tahirah"],
}

# Urdu script renderings for common tokens. A full name is rendered in Urdu
# only when every one of its tokens has an entry here.
URDU: dict[str, str] = {
    "Muhammad": "محمد", "Ahmed": "احمد", "Ali": "علی", "Hassan": "حسن",
    "Hussain": "حسین", "Usman": "عثمان", "Bilal": "بلال", "Imran": "عمران",
    "Asif": "آصف", "Tariq": "طارق", "Khalid": "خالد", "Faisal": "فیصل",
    "Kamran": "کامران", "Shahid": "شاہد", "Zafar": "ظفر", "Rashid": "رشید",
    "Naveed": "نوید", "Salman": "سلمان", "Adnan": "عدنان", "Farhan": "فرحان",
    "Junaid": "جنید", "Kashif": "کاشف", "Nadeem": "ندیم", "Saeed": "سعید",
    "Waqar": "وقار", "Yasir": "یاسر", "Zubair": "زبیر", "Arif": "عارف",
    "Javed": "جاوید", "Iqbal": "اقبال", "Akram": "اکرم", "Aslam": "اسلم",
    "Raza": "رضا", "Umar": "عمر", "Qasim": "قاسم", "Anwar": "انور",
    "Riaz": "ریاض", "Sohail": "سہیل", "Nawaz": "نواز", "Irfan": "عرفان",
    "Khan": "خان", "Malik": "ملک", "Butt": "بٹ", "Sheikh": "شیخ",
    "Chaudhry": "چوہدری", "Qureshi": "قریشی", "Siddiqui": "صدیقی",
    "Awan": "اعوان", "Abbasi": "عباسی", "Raja": "راجہ", "Mirza": "مرزا",
    "Baig": "بیگ", "Ansari": "انصاری", "Hashmi": "ہاشمی", "Gilani": "گیلانی",
    "Bhatti": "بھٹی", "Dar": "ڈار", "Lodhi": "لودھی", "Niazi": "نیازی",
    "Mughal": "مغل", "Shah": "شاہ", "Naqvi": "نقوی", "Rizvi": "رضوی",
    "Zaidi": "زیدی", "Kazmi": "کاظمی", "Abbas": "عباس", "Akhtar": "اختر",
    "Ayesha": "عائشہ", "Fatima": "فاطمہ", "Zainab": "زینب",
    "Maryam": "مریم", "Khadija": "خدیجہ", "Sana": "ثنا", "Hira": "حرا",
    "Saba": "صبا", "Bushra": "بشریٰ", "Samina": "ثمینہ", "Tahira": "طاہرہ",
    "Begum": "بیگم", "Bibi": "بی بی", "Sadia": "سعدیہ", "Farah": "فرح",
}

MALE_HONORIFICS = ["Mr.", "Haji", "Mian", "Dr.", "Engr."]
FEMALE_HONORIFICS = ["Mrs.", "Ms.", "Dr.", "Mst."]


def pick_name_tokens(rng: random.Random, gender: str, family: str | None = None) -> list[str]:
    """Canonical token list for a new person, e.g. ["Muhammad", "Asif", "Khan"]."""
    fam = family or rng.choice(FAMILY)
    if gender == "M":
        first = rng.choice(MALE_FIRST)
        tokens = [first, fam]
        if rng.random() < 0.45:
            tokens = ["Muhammad"] + tokens
    else:
        first = rng.choice(FEMALE_FIRST)
        tokens = [first, fam]
        if rng.random() < 0.15:
            tokens = [first, rng.choice(["Bibi", "Begum"])]
    return tokens


def render_roman(tokens: list[str], rng: random.Random, gender: str = "M") -> str:
    """Render tokens as a Roman-script name with realistic variation."""
    out = []
    for tok in tokens:
        choices = [tok] + VARIANTS.get(tok, [])
        weights = [3.0] + [1.0] * (len(choices) - 1)
        out.append(rng.choices(choices, weights=weights)[0])
    name = " ".join(out)
    r = rng.random()
    if r < 0.07:
        hon = rng.choice(MALE_HONORIFICS if gender == "M" else FEMALE_HONORIFICS)
        name = f"{hon} {name}"
    elif r < 0.12:
        name = name.upper()
    return name


def render_urdu(tokens: list[str]) -> str | None:
    """Urdu-script rendering, or None if any token lacks a mapping."""
    if all(t in URDU for t in tokens):
        return " ".join(URDU[t] for t in tokens)
    return None


def render_name(tokens: list[str], rng: random.Random, gender: str = "M",
                urdu_prob: float = 0.0) -> str:
    if urdu_prob > 0 and rng.random() < urdu_prob:
        urdu = render_urdu(tokens)
        if urdu:
            return urdu
    return render_roman(tokens, rng, gender)
