# Pitch — Graph AI for Broadening the National Tax Net

*Material for Round 2 (Value Proposition · Market Viability · Pitch & Delivery).
Use as the basis for slides and the live demo.*

---

## 1. The problem (the stakes)

Pakistan's tax-to-GDP ratio is among the lowest in the world (~9–10%). The cause
isn't a lack of *data* — the state already knows who owns the 3000cc Land Cruiser,
who pays Rs 400k/month in electricity, who flies abroad six times a year. The
problem is that this data sits in **disconnected silos** (Excise, FBR, DISCOs,
FIA travel, land registries) that are never linked intelligently. High-net-worth
non-filers hide in the gaps between databases.

**One-line problem:** the wealth is visible; the *links* are not.

## 2. The solution & value proposition

A system that **links fragmented civic databases into one knowledge graph**,
resolves scattered records into real individuals (despite Urdu/English name
variants and dirty CNICs), and flags those whose **lifestyle implies far more
income than they declare** — with a plain-language audit trail for every flag.

**Three things that make it valuable, not just clever:**

1. **It works on the data that already exists.** No new collection, no new law —
   just intelligent linkage of records the state already holds.
2. **Every flag is explainable and audit-ready.** "Declares Rs 0, owns Rs 5cr in
   property, pays Rs 400k/month electricity" — an officer can act on it tomorrow.
   This is the difference between a research model and a deployable GovTech tool.
3. **It catches what humans and spreadsheets cannot: hidden networks.** Our
   headline capability — detecting **benami/proxy ownership**, where a wealthy
   evader registers assets under frontmen. Their own books look clean; only the
   graph exposes them.

## 3. Differentiation — why this beats existing approaches

| Existing approach | Limitation | Our approach |
|---|---|---|
| Manual cross-checking / spreadsheets | Can't scale; misses cross-silo links | Automated entity resolution across 5+ silos |
| Rule-based flags ("bill > X") | Rigid, gameable, no networks | Unsupervised ML learns the population; graph finds networks |
| Black-box ML scoring | Not auditable → not actionable for tax law | Explainable audit trail in PKR per flag |
| Tabular fraud models | **Blind to proxy/benami structures** | **GNN catches ~93% of proxy-using principals vs ~4% for tabular** |

That last row is the moat. We *measured* it: on evaders who hide wealth through
proxies, a tabular model recovers **~4%**; graph message-passing recovers **~93%**.

## 4. Proof it works (validation)

Built end-to-end on synthetic-but-realistic Pakistani civic data with a strict
"wall" (the detector never sees the answer key):

Run at 10,000 people (~31,500 records), full pipeline in ~26 seconds:

- **Entity resolution:** precision 0.99, recall 0.95 (handles Urdu/English
  transliteration + 22% corrupted/missing CNICs).
- **Evasion detection:** average precision 0.71 (≈2.3× the base rate),
  precision@25% ≈ 0.75.
- **Proxy networks:** **Rs 364 crore** in hidden assets surfaced across 233
  detected proxy networks; principal recall **0.04 → 0.93** once the graph is used.
- **Estimated tax-base gap** on flagged individuals: Rs 206 crore.
- **25 automated tests**; near-linear scaling demonstrated to 18k+ records.

## 5. Market viability & growth

**Primary market — sovereign GovTech (FBR / provincial excise).** Directly
attacks the tax-gap mandate. Even a 1-point tax-to-GDP improvement is worth
billions of dollars annually; the system pays for itself on a handful of
recovered high-net-worth cases.

**Adjacent market — banking KYC/AML (the same engine).** Entity resolution +
graph anomaly detection over financial silos is exactly what banks need for
SBP-mandated AML compliance, beneficial-ownership discovery, and fraud rings. The
proxy/benami detection is structurally identical to money-mule and shell-network
detection. This doubles the addressable market with zero re-architecture.

**Why it can scale & be adopted:**
- Runs on existing data; integrates as a read-only analytics layer over current
  databases — low operational barrier, no disruption to live systems.
- Blocking-based design ports to Neo4j + Spark for national scale (demonstrated
  sub-quadratic; see README).
- Explainability makes it legally actionable and builds institutional trust —
  the usual blocker for AI in government.

**Growth path:** pilot on one province's excise + DISCO data → expand silos
(FIA travel, land registry, bank feeds) → each new silo compounds the graph's
power → license the same engine to the banking sector for AML.

## 6. The 3-minute demo script

1. **Hook (20s):** "The state already knows who owns the Land Cruiser and pays
   the Rs 400k electricity bill. It just never connects the dots. We do."
2. **Pipeline (40s):** show the dashboard KPIs — entities analysed, estimated
   tax-base gap, proxy networks detected.
3. **A simple catch (40s):** open a top-flagged non-filer; read the audit trail
   aloud — concrete, explainable, audit-ready.
4. **The killer demo (60s):** switch to the Proxy/Benami tab. "This man files a
   clean return. A spreadsheet clears him. But watch —" expand his network graph:
   the proxies light up holding crores in undeclared assets. "Tabular analysis
   catches ~4% of these. Our graph AI catches ~93%."
5. **Close (20s):** validation numbers + "works on existing data, explainable,
   and the same engine does bank AML. This is deployable, not a science project."

## 7. The one slide to remember

> **Tabular analysis nabs the frontmen. Graph AI nabs the kingpin.**
> Built on data the state already has. Every flag explainable. Same engine
> powers bank AML. Pilot-ready.
