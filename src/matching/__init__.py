"""Multi-tier entity-resolution cascade (ported from the hackathon-tax-net donor).

Tier 1  — unsupervised Fellegi-Sunter probabilistic matching over blocked pairs.
Tier 2  — multilingual sentence-embedding rescue for cross-script / weak-name pairs.
Tier 4  — collective resolution over the identifier graph (PROMOTE + SPLIT).

The donor's Tier 3 (local-LLM adjudication) is intentionally NOT ported.

Every module reads ONLY data/observable/*.csv (+ its own data/er_output/ artifacts)
— never the sealed ground truth. A DOB evidence band (absent in the donor) is added
throughout: it is tax-net's highest-signal disambiguator and the recall lever.
"""
