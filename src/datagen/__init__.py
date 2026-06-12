"""Synthetic Pakistani civic-data generator (ported from the hackathon-tax-net
donor and adapted to tax-net's observable + ground-truth schema).

Richness this brings over the original flat generator: per-registry Urdu script
rendering, masked/typo'd CNICs, phone identifiers with format variation, address
rendering variation, real vehicle makes/cc, and hand-seeded edge personas. Every
record is still enriched with DOB + father's name (the entity-resolution recall
lever) and the population carries tax-net's latent labels behind the wall.
"""
