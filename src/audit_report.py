"""
Stage 4e — Explainable audit notice: why was this entity flagged?

Adapted from the donor's AuditBuilder to tax-net's artifacts. For a resolved
entity it assembles, all deterministically from pipeline outputs:

  1. ML assessment  - the HEADLINE ensemble Tax Compliance Deviation Score
                       (IF + GNN), its per-model breakdown, and the percentile
                       reasons the detector produced.
  2. identity        - the records linked into this entity and, per link, which
                       tier decided it and on what evidence (from match_evidence).
  3. lifestyle floors- the explainable rule-based cross-check (rule_floors):
                       each implied-income floor with its stated assumption and
                       the resulting tax-gap arithmetic.
  4. observations    - ghost status, proxy/benami household notes.

Output: structured dict, Markdown narrative (bilingual header), and an FBR-style
PDF notice (Urdu header rendered via a Windows Arabic-capable font when present,
English-only otherwise — generation never fails).

Reads ONLY data/resolved/* + data/observable/* — never ground truth.
Writes data/audit/{entity_id}.{json,md,pdf}.
"""

import os
import json

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(ROOT, "data", "resolved")
OBS_DIR = os.path.join(ROOT, "data", "observable")
AUDIT_DIR = os.path.join(ROOT, "data", "audit")

URDU_HEADER = "ٹیکس تعمیل جائزہ نوٹس"  # "tax compliance review notice"

# observable file -> id column (records carry different id column names)
_OBS_IDCOL = {
    "vehicles": "record_id", "real_estate": "record_id", "travel": "record_id",
    "utilities": "account_id", "tax_returns": "return_id",
}


def _safe_int(v, default=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


class AuditBuilder:
    def __init__(self, data_dir: str | None = None) -> None:
        res = os.path.join(data_dir, "resolved") if data_dir else RES_DIR
        obs = os.path.join(data_dir, "observable") if data_dir else OBS_DIR
        self.mentions = pd.read_csv(os.path.join(res, "mentions.csv"), dtype=str)
        self.mentions["entity_id"] = self.mentions["entity_id"].astype(int)

        self.scores = pd.read_csv(os.path.join(res, "entity_scores.csv"))
        self.scores["entity_id"] = self.scores["entity_id"].astype(int)
        self.score_by = {int(r.entity_id): r._asdict()
                         for r in self.scores.itertuples()}

        self.floors = pd.read_csv(os.path.join(res, "entity_floors.csv"))
        self.floors["entity_id"] = self.floors["entity_id"].astype(int)
        self.floor_by = {int(r.entity_id): r._asdict()
                         for r in self.floors.itertuples()}

        with open(os.path.join(res, "factors.json"), encoding="utf-8") as fh:
            self.factors = json.load(fh)

        ev_path = os.path.join(res, "match_evidence.csv")
        self.evidence = (pd.read_csv(ev_path, dtype=str)
                         if os.path.exists(ev_path) and os.path.getsize(ev_path)
                         else pd.DataFrame(columns=["record_a", "record_b",
                                                    "decided_by", "evidence", "posterior"]))

        self.obs_fields = {}
        for src, idcol in _OBS_IDCOL.items():
            df = pd.read_csv(os.path.join(obs, f"{src}.csv"), dtype=str)
            for row in df.to_dict("records"):
                self.obs_fields[row[idcol]] = {k: v for k, v in row.items()
                                               if isinstance(v, str) and v and v != "nan"}

    def build(self, entity_id) -> dict:
        eid = int(entity_id)
        recs = self.mentions[self.mentions.entity_id == eid]
        records = [{"record_id": r.record_id, "registry": r.source,
                    "name": r.raw_name, "fields": self.obs_fields.get(r.record_id, {})}
                   for r in recs.itertuples()]
        rec_ids = set(recs.record_id)

        links = []
        if len(self.evidence):
            sub = self.evidence[self.evidence.record_a.isin(rec_ids)
                                & self.evidence.record_b.isin(rec_ids)]
            for p in sub.itertuples():
                link = {"records": f"{p.record_a} <-> {p.record_b}",
                        "decided_by": p.decided_by or "tier1",
                        "evidence": p.evidence or ""}
                if isinstance(p.posterior, str) and p.posterior:
                    try:
                        link["confidence"] = float(p.posterior)
                    except ValueError:
                        pass
                links.append(link)

        floor = self.floor_by.get(eid, {})
        fac = self.factors.get(str(eid), {"factors": [], "notes": []})
        srow = self.score_by.get(eid, {})

        try:
            ml_reasons = json.loads(srow.get("audit_trail", "[]"))
        except (TypeError, ValueError):
            ml_reasons = []
        headline = srow.get("deviation_score_combined", srow.get("deviation_score", 0))

        return {
            "entity_id": eid,
            "display_name": floor.get("display_name", f"Entity {eid}"),
            "city": floor.get("city", ""),
            "records": records,
            "links": links,
            "lifestyle_factors": fac["factors"],
            "notes": fac["notes"],
            "ml": {
                "deviation_score": _safe_int(round(float(headline or 0))),
                "deviation_score_if": _safe_int(round(float(srow.get("deviation_score", 0) or 0))),
                "gnn_score": _safe_int(round(float(srow.get("gnn_score", 0) or 0))),
                "reasons": ml_reasons,
            },
            "cross_check": {
                "estimated_income_pkr": _safe_int(floor.get("estimated_income_pkr", 0)),
                "declared_income_pkr": _safe_int(floor.get("declared_income_pkr", 0)),
                "filer_status": floor.get("filer_status", "absent"),
                "tax_paid_pkr": _safe_int(floor.get("tax_paid_pkr", 0)),
                "expected_tax_pkr": _safe_int(floor.get("expected_tax_pkr", 0)),
                "tax_gap_pkr": _safe_int(floor.get("tax_gap_pkr", 0)),
                "rule_deviation_score": _safe_int(floor.get("rule_deviation_score", 0)),
                "band": floor.get("band", "none"),
            },
        }

    def markdown(self, audit: dict) -> str:
        ml, cc = audit["ml"], audit["cross_check"]
        regs = len({r["registry"] for r in audit["records"]})
        lines = [
            f"# Tax Compliance Review Notice / {URDU_HEADER}",
            "",
            f"**Entity:** {audit['entity_id']} — {audit['display_name']}, {audit['city']}",
            f"**Tax Compliance Deviation Score (production, IF+GNN ensemble): "
            f"{ml['deviation_score']}/100**",
            f"  - tabular (Isolation Forest): {ml['deviation_score_if']}/100  ·  "
            f"relational (GNN): {ml['gnn_score']}/100",
            "",
            "## 1. Why the model flagged this profile",
        ]
        if ml["reasons"]:
            for r in ml["reasons"]:
                lines.append(f"- {r}")
        else:
            lines.append("- (no percentile reasons recorded)")
        lines += [
            "",
            "## 2. Identity resolution",
            f"Assembled from {len(audit['records'])} records across {regs} "
            "independent registries. Every link below was decided by the stated "
            "tier on the stated evidence — only public-registry data was used.",
            "",
        ]
        for r in audit["records"]:
            lines.append(f"- `{r['record_id']}` ({r['registry']}): {r['name']}")
        if audit["links"]:
            lines += ["", "### Link evidence"]
            for l in audit["links"]:
                conf = f" (p={l['confidence']:.3f})" if "confidence" in l else ""
                lines.append(f"- {l['records']} — {l['decided_by']}{conf}: {l['evidence']}")
        else:
            lines += ["", "_Per-link evidence table not available._"]
        lines += ["", "## 3. Explainable cross-check — lifestyle-implied income floors",
                  "_Secondary, deterministic sanity check (not the headline score)._"]
        for f in audit["lifestyle_factors"]:
            lines.append(f"- **{f['factor']}**: implies ≥ {f['implied_income']:,} "
                         f"PKR/yr — {f['detail']}")
        lines += [
            "",
            f"- Estimated income (strongest floor): **{cc['estimated_income_pkr']:,} PKR/yr**",
            f"- Declared income: {cc['declared_income_pkr']:,} PKR ({cc['filer_status']})",
            f"- Expected tax at estimate: {cc['expected_tax_pkr']:,} PKR  ·  "
            f"tax paid: {cc['tax_paid_pkr']:,} PKR",
            f"- **Lifestyle-implied tax gap: {cc['tax_gap_pkr']:,} PKR** "
            f"(rule band: {cc['band']})",
        ]
        if audit["notes"]:
            lines += ["", "## 4. Observations"]
            for n in audit["notes"]:
                lines.append(f"- {n}")
        lines += [
            "", "---",
            "Generated automatically by TaxNet on synthetic data. The headline "
            "score is an unsupervised ML signal; the lifestyle floors are review "
            "grounds derived from stated assumptions, not a determination of liability.",
        ]
        return "\n".join(lines)

    def pdf(self, audit: dict, path) -> None:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.lib import colors
        from reportlab.platypus import (HRFlowable, Paragraph, SimpleDocTemplate,
                                        Spacer, Table, TableStyle)

        styles = getSampleStyleSheet()
        ml, cc = audit["ml"], audit["cross_check"]
        story = [Paragraph("FEDERAL BOARD OF REVENUE (DEMO)", styles["Title"]),
                 Paragraph("Tax Compliance Review Notice", styles["Heading2"])]

        urdu_font = _register_urdu_font()
        shaped = _shape_urdu(URDU_HEADER) if urdu_font else None
        if urdu_font and shaped:
            story.append(Paragraph(
                shaped, ParagraphStyle("urdu", parent=styles["Heading2"],
                                       fontName=urdu_font, alignment=2)))

        story += [
            HRFlowable(width="100%"), Spacer(1, 6),
            Paragraph(f"<b>Reference:</b> {audit['entity_id']} &nbsp;&nbsp; "
                      f"<b>Name:</b> {audit['display_name']} &nbsp;&nbsp; "
                      f"<b>City:</b> {audit['city']}", styles["Normal"]),
            Spacer(1, 8),
            Table([["Deviation score (IF+GNN ensemble)", f"{ml['deviation_score']}/100"],
                   ["  tabular (Isolation Forest)", f"{ml['deviation_score_if']}/100"],
                   ["  relational (GNN)", f"{ml['gnn_score']}/100"]],
                  colWidths=[80 * mm, 80 * mm],
                  style=TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                                    ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                                    ("FONTSIZE", (0, 0), (-1, -1), 9)])),
            Spacer(1, 10),
            Paragraph("Why the model flagged this profile", styles["Heading3"]),
        ]
        for r in (ml["reasons"] or ["(no percentile reasons recorded)"]):
            story.append(Paragraph(f"&bull; {r}", styles["Normal"]))

        story += [Spacer(1, 8),
                  Paragraph("Explainable cross-check — lifestyle-implied income floors",
                            styles["Heading3"])]
        for f in audit["lifestyle_factors"]:
            story.append(Paragraph(
                f"&bull; <b>{f['factor']}</b>: implies at least "
                f"{f['implied_income']:,} PKR/yr — {f['detail']}", styles["Normal"]))
        story.append(Table(
            [["Estimated income", f"{cc['estimated_income_pkr']:,} PKR/yr"],
             ["Declared income", f"{cc['declared_income_pkr']:,} PKR ({cc['filer_status']})"],
             ["Expected tax", f"{cc['expected_tax_pkr']:,} PKR"],
             ["Tax paid", f"{cc['tax_paid_pkr']:,} PKR"],
             ["Lifestyle-implied tax gap", f"{cc['tax_gap_pkr']:,} PKR ({cc['band']})"]],
            colWidths=[60 * mm, 100 * mm],
            style=TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                              ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                              ("FONTSIZE", (0, 0), (-1, -1), 9)])))
        if audit["notes"]:
            story.append(Paragraph("Observations", styles["Heading3"]))
            for n in audit["notes"]:
                story.append(Paragraph(f"&bull; {n}", styles["Normal"]))
        story += [
            Spacer(1, 8),
            Paragraph(f"Identity assembled from {len(audit['records'])} registry "
                      f"records across independent silos.", styles["Normal"]),
            Spacer(1, 12), HRFlowable(width="100%"),
            Paragraph("Generated automatically by TaxNet on synthetic data. The "
                      "headline score is an unsupervised ML signal; lifestyle floors "
                      "are review grounds, not determinations of liability.",
                      styles["Italic"]),
        ]
        SimpleDocTemplate(str(path), pagesize=A4, topMargin=18 * mm,
                          bottomMargin=18 * mm).build(story)


def _register_urdu_font():
    """Register a Windows Arabic-capable TTF for the Urdu header; None if absent."""
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except Exception:
        return None
    for name, fpath in [("UrduArabic", r"C:\Windows\Fonts\tahoma.ttf"),
                        ("UrduArabic", r"C:\Windows\Fonts\arial.ttf")]:
        if os.path.exists(fpath):
            try:
                pdfmetrics.registerFont(TTFont(name, fpath))
                return name
            except Exception:
                continue
    return None


def _shape_urdu(text: str):
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        return None


def make_audits(top_n: int = 25, data_dir: str | None = None) -> dict:
    res = os.path.join(data_dir, "resolved") if data_dir else RES_DIR
    audit_dir = os.path.join(data_dir, "audit") if data_dir else AUDIT_DIR
    os.makedirs(audit_dir, exist_ok=True)

    builder = AuditBuilder(data_dir)
    rank_col = ("deviation_score_combined" if "deviation_score_combined"
                in builder.scores.columns else "deviation_score")
    top = builder.scores.nlargest(top_n, rank_col)["entity_id"].tolist()

    written = 0
    for eid in top:
        audit = builder.build(eid)
        with open(os.path.join(audit_dir, f"{eid}.json"), "w", encoding="utf-8") as fh:
            json.dump(audit, fh, indent=1, ensure_ascii=False)
        with open(os.path.join(audit_dir, f"{eid}.md"), "w", encoding="utf-8") as fh:
            fh.write(builder.markdown(audit))
        try:
            builder.pdf(audit, os.path.join(audit_dir, f"{eid}.pdf"))
        except Exception as e:
            print(f"  [audit] PDF for {eid} skipped ({type(e).__name__}: {e})")
        written += 1
    print(f"  wrote {written} audit notices (json/md/pdf) to data/audit/")
    return {"audits_written": written, "ranked_by": rank_col}


if __name__ == "__main__":
    make_audits()
