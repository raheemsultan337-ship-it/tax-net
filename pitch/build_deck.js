/* Build the Round-2 pitch deck for tax-net (CUST Hackathon 2026, Problem #2).
   Numbers are the CURRENT verified 50k-scale results, not PITCH.md's 10k-era ones.
   Run:  NODE_PATH=<global node_modules> node build_deck.js                      */
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const {
  FaCar, FaBolt, FaPlane, FaFileInvoiceDollar, FaDatabase, FaFingerprint,
  FaProjectDiagram, FaTachometerAlt, FaClipboardList, FaBalanceScale,
  FaUserSecret, FaShieldAlt, FaCheckCircle, FaLandmark, FaUniversity, FaLink,
  FaSearchDollar, FaLanguage, FaTree, FaCogs,
} = require("react-icons/fa");

const GREEN_D = "0B3D2E";   // deep green (dark slides, headings)
const GREEN_M = "1D6F50";   // mid green (icon circles)
const GREEN_T = "EAF3EE";   // light green tint (cards)
const GOLD    = "D9A441";   // accent
const INK     = "1A2B24";   // body text on light
const MUTED   = "5C6F66";   // secondary text on light
const PALE    = "B8CCC2";   // secondary text on dark
const CARD_DK = "11503C";   // card fill on dark slides

const HEAD = "Cambria";
const BODY = "Calibri";

function renderIconSvg(Icon, color, size = 256) {
  return ReactDOMServer.renderToStaticMarkup(
    React.createElement(Icon, { color, size: String(size) }));
}
async function iconPng(Icon, colorHex) {
  const svg = renderIconSvg(Icon, "#" + colorHex);
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}
const shadow = () => ({ type: "outer", color: "000000", blur: 7, offset: 2, angle: 45, opacity: 0.14 });

function addLine(slide, pres, x1, y1, x2, y2, opts) {
  const x = Math.min(x1, x2), y = Math.min(y1, y2);
  const w = Math.abs(x2 - x1), h = Math.abs(y2 - y1);
  const flipV = ((x2 - x1) > 0) !== ((y2 - y1) > 0);
  slide.addShape(pres.shapes.LINE, { x, y, w, h, flipV, line: opts });
}

async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";
  pres.title = "Graph AI for Broadening the National Tax Net";
  pres.author = "tax-net team";

  // pre-render icons
  const ic = {};
  const want = {
    car: [FaCar, "FFFFFF"], bolt: [FaBolt, "FFFFFF"], plane: [FaPlane, "FFFFFF"],
    invoice: [FaFileInvoiceDollar, "FFFFFF"], db: [FaDatabase, "FFFFFF"],
    finger: [FaFingerprint, "FFFFFF"], graph: [FaProjectDiagram, "FFFFFF"],
    gauge: [FaTachometerAlt, "FFFFFF"], clip: [FaClipboardList, "FFFFFF"],
    scale: [FaBalanceScale, "FFFFFF"], secret: [FaUserSecret, "FFFFFF"],
    shield: [FaShieldAlt, GREEN_M], check: [FaCheckCircle, GREEN_M],
    landmark: [FaLandmark, "FFFFFF"], bank: [FaSearchDollar, "FFFFFF"],
    lang: [FaLanguage, "FFFFFF"], tree: [FaTree, "FFFFFF"], cogs: [FaCogs, "FFFFFF"],
    link: [FaLink, GOLD], shieldGold: [FaShieldAlt, GOLD],
    clipGold: [FaClipboardList, GOLD], bankGold: [FaUniversity, GOLD],
  };
  for (const [k, [Icon, color]] of Object.entries(want)) ic[k] = await iconPng(Icon, color);

  // helper: icon inside a colored circle
  function iconCircle(slide, key, cx, cy, d, circleColor) {
    slide.addShape(pres.shapes.OVAL, { x: cx, y: cy, w: d, h: d, fill: { color: circleColor } });
    const pad = d * 0.26;
    slide.addImage({ data: ic[key], x: cx + pad, y: cy + pad, w: d - 2 * pad, h: d - 2 * pad });
  }

  /* ---------------- Slide 1 — title ---------------- */
  let s = pres.addSlide();
  s.background = { color: GREEN_D };
  // network constellation, right side
  const nodes = [
    [7.05, 1.35, 0.30], [8.45, 1.00, 0.22], [9.10, 2.10, 0.26],
    [7.70, 2.55, 0.40, GOLD], [8.85, 3.55, 0.24], [6.95, 3.80, 0.22], [8.05, 4.55, 0.18],
  ];
  const edges = [[0, 3], [1, 3], [2, 3], [3, 4], [3, 5], [4, 6], [5, 6], [1, 2]];
  for (const [a, b] of edges) {
    const [ax, ay, ad] = nodes[a], [bx, by, bd] = nodes[b];
    addLine(s, pres, ax + ad / 2, ay + ad / 2, bx + bd / 2, by + bd / 2,
      { color: "3E7A63", width: 1.25 });
  }
  for (const [x, y, d, c] of nodes)
    s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: c || "F2F6F4" } });
  s.addText("Graph AI for broadening\nthe national tax net", {
    x: 0.6, y: 1.30, w: 6.1, h: 1.75, fontFace: HEAD, fontSize: 38, bold: true,
    color: "FFFFFF", lineSpacing: 44,
  });
  s.addText("The wealth is visible. The links are not.", {
    x: 0.62, y: 3.15, w: 6.0, h: 0.5, fontFace: BODY, fontSize: 20, italic: true, color: GOLD,
  });
  s.addText("CUST Hackathon 2026  ·  Problem #2 (Ciklum)  ·  FinTech · knowledge graphs · fraud detection", {
    x: 0.62, y: 4.95, w: 8.8, h: 0.35, fontFace: BODY, fontSize: 11, color: PALE,
  });
  s.addNotes("Hook (20s): The state already knows who owns the Land Cruiser and who pays the Rs 400k electricity bill. It just never connects the dots. We built the system that does.");

  /* ---------------- Slide 2 — problem ---------------- */
  s = pres.addSlide();
  s.background = { color: "FFFFFF" };
  s.addText("Billions hiding between databases", {
    x: 0.5, y: 0.32, w: 9.0, h: 0.6, fontFace: HEAD, fontSize: 30, bold: true, color: GREEN_D,
  });
  s.addText([
    { text: "Pakistan's tax-to-GDP ratio is among the lowest in the world — and not for lack of data. The state already records who owns the 3000cc SUV, who pays Rs 400,000 a month in electricity, who flies abroad six times a year.", options: { breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "But Excise, FBR, DISCOs, FIA and the land registries never talk to each other. High-net-worth non-filers live in the gaps between databases.", options: {} },
  ], { x: 0.5, y: 1.10, w: 4.55, h: 2.4, fontFace: BODY, fontSize: 14, color: INK, valign: "top" });
  s.addText("~9%", {
    x: 0.5, y: 3.55, w: 2.2, h: 1.05, fontFace: HEAD, fontSize: 62, bold: true, color: GREEN_D, margin: 0,
  });
  s.addText("tax-to-GDP ratio —\namong the world's lowest", {
    x: 2.75, y: 3.72, w: 2.4, h: 0.8, fontFace: BODY, fontSize: 12, color: MUTED, valign: "middle",
  });
  // case card
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 5.45, y: 1.05, w: 4.05, h: 3.95, rectRadius: 0.09,
    fill: { color: GREEN_T }, shadow: shadow(),
  });
  s.addText("One citizen, four databases", {
    x: 5.75, y: 1.22, w: 3.5, h: 0.4, fontFace: BODY, fontSize: 15, bold: true, color: GREEN_D, margin: 0,
  });
  const rows = [
    ["car", "3000cc Land Cruiser", "Excise vehicle registry"],
    ["bolt", "Rs 400,000 / month", "DISCO electricity billing"],
    ["plane", "6 international trips", "FIA travel logs"],
    ["invoice", "Declares: Rs 0", "FBR tax return"],
  ];
  rows.forEach(([key, big, small], i) => {
    const y = 1.78 + i * 0.78;
    iconCircle(s, key, 5.75, y, 0.5, i === 3 ? GOLD : GREEN_M);
    s.addText(big, { x: 6.42, y: y - 0.04, w: 3.0, h: 0.32, fontFace: BODY, fontSize: 14, bold: true, color: i === 3 ? "8A5A10" : INK, margin: 0 });
    s.addText(small, { x: 6.42, y: y + 0.26, w: 3.0, h: 0.28, fontFace: BODY, fontSize: 11, color: MUTED, margin: 0 });
  });
  s.addText("Every record above exists today. No system connects them.", {
    x: 5.45, y: 5.12, w: 4.05, h: 0.35, fontFace: BODY, fontSize: 11, italic: true, color: MUTED, align: "center",
  });
  s.addNotes("The cause is not missing data — it is disconnection. Name the silos. End on: the wealth is visible, the links are not.");

  /* ---------------- Slide 3 — pipeline ---------------- */
  s = pres.addSlide();
  s.background = { color: "FFFFFF" };
  s.addText("From raw silos to an audit-ready flag", {
    x: 0.5, y: 0.30, w: 9.0, h: 0.6, fontFace: HEAD, fontSize: 30, bold: true, color: GREEN_D,
  });
  const steps = [
    ["db", "1 · Ingest five silos", "Vehicles, property, utilities, travel, tax returns — 195k messy records"],
    ["finger", "2 · Resolve identities", "Unsupervised matching survives Urdu/English names and dirty CNICs"],
    ["graph", "3 · Build the graph", "People linked to assets and households — 53,586 entities"],
    ["gauge", "4 · Score deviation 0–100", "Isolation Forest + graph neural network, fully unsupervised"],
    ["clip", "5 · Explain every flag", "Plain-PKR audit trail and a bilingual Urdu/English PDF notice"],
    ["scale", "6 · Prove it honestly", "Held-out scorecard — the detector never sees the answer key"],
  ];
  steps.forEach(([key, title, body], i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = 0.5 + col * 3.10, y = 1.10 + row * 1.95;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y, w: 2.90, h: 1.78, rectRadius: 0.08, fill: { color: GREEN_T }, shadow: shadow(),
    });
    iconCircle(s, key, x + 0.20, y + 0.18, 0.46, GREEN_M);
    s.addText(title, { x: x + 0.80, y: y + 0.22, w: 2.02, h: 0.42, fontFace: BODY, fontSize: 13.5, bold: true, color: GREEN_D, margin: 0, valign: "middle" });
    s.addText(body, { x: x + 0.20, y: y + 0.78, w: 2.52, h: 0.92, fontFace: BODY, fontSize: 11.5, color: INK, margin: 0, valign: "top" });
  });
  s.addText("One command end-to-end  ·  50,000-citizen simulation  ·  ~20 minutes on a laptop", {
    x: 0.5, y: 5.12, w: 9.0, h: 0.35, fontFace: BODY, fontSize: 11.5, italic: true, color: MUTED, align: "center",
  });
  s.addNotes("Pipeline (40s): five registries in, one knowledge graph, one 0-100 deviation score out — and every flag arrives with its own audit trail. Everything unsupervised: no labels, because in the real world nobody hands you a list of evaders.");

  /* ---------------- Slide 4 — the AI inside ---------------- */
  s = pres.addSlide();
  s.background = { color: "FFFFFF" };
  s.addText("The AI inside — four models, one score", {
    x: 0.5, y: 0.30, w: 9.0, h: 0.6, fontFace: HEAD, fontSize: 30, bold: true, color: GREEN_D,
  });
  const models = [
    ["finger", "Fellegi–Sunter probabilistic matcher",
      "Self-calibrating record linkage — learns evidence weights (CNIC, DOB, father's name) from the data itself, no labels"],
    ["lang", "Multilingual sentence embeddings",
      "MiniLM transformer maps Urdu and English spellings of the same name to the same point in vector space"],
    ["tree", "Isolation Forest",
      "Unsupervised anomaly detection on footprint-vs-declared ratios — rich-but-honest filers stay inliers"],
    ["graph", "GraphSAGE autoencoder (GNN)",
      "Message passing over the knowledge graph — flags benami networks by their shape, not their books"],
  ];
  models.forEach(([key, title, desc], i) => {
    const y = 1.12 + i * 0.97;
    iconCircle(s, key, 0.5, y, 0.5, GREEN_M);
    s.addText(title, { x: 1.18, y: y - 0.05, w: 4.55, h: 0.34, fontFace: BODY, fontSize: 13.5, bold: true, color: GREEN_D, margin: 0 });
    s.addText(desc, { x: 1.18, y: y + 0.29, w: 4.55, h: 0.62, fontFace: BODY, fontSize: 11, color: INK, margin: 0, valign: "top" });
  });
  // stack card
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.05, y: 1.12, w: 3.45, h: 3.78, rectRadius: 0.09, fill: { color: GREEN_T }, shadow: shadow() });
  iconCircle(s, "cogs", 6.30, 1.32, 0.46, GREEN_M);
  s.addText("Engineering stack", { x: 6.90, y: 1.36, w: 2.5, h: 0.38, fontFace: BODY, fontSize: 14, bold: true, color: GREEN_D, margin: 0 });
  s.addText([
    { text: "Python · pandas · scikit-learn · NetworkX knowledge graph", options: { bullet: true, breakLine: true } },
    { text: "PyTorch Geometric (GNN) · sentence-transformers", options: { bullet: true, breakLine: true } },
    { text: "Streamlit dashboard · ReportLab bilingual PDF notices", options: { bullet: true, breakLine: true } },
    { text: "48 pytest tests incl. a wall test on label isolation", options: { bullet: true, breakLine: true } },
    { text: "Scale path: blocked matching (near-linear) → Neo4j + Spark", options: { bullet: true } },
  ], { x: 6.30, y: 1.92, w: 3.05, h: 2.85, fontFace: BODY, fontSize: 11, color: INK, paraSpaceAfter: 6, valign: "top" });
  s.addText("An ensemble blends both detectors into one 0–100 deviation score — quantile-aligned so the audit threshold keeps its meaning.", {
    x: 0.5, y: 5.12, w: 9.0, h: 0.35, fontFace: BODY, fontSize: 11.5, italic: true, color: MUTED, align: "center",
  });
  s.addNotes("If judges want depth: the matcher calibrates itself with no labels, embeddings solve Urdu/English, the Isolation Forest works on ratios so wealth alone is never a flag, and the GNN autoencoder catches network shapes. All four are unsupervised - no answer key anywhere in training.");

  /* ---------------- Slide 5 — the moat ---------------- */
  s = pres.addSlide();
  s.background = { color: GREEN_D };
  s.addText("The moat: catch the kingpin, not the frontman", {
    x: 0.5, y: 0.32, w: 9.0, h: 0.6, fontFace: HEAD, fontSize: 28, bold: true, color: "FFFFFF",
  });
  iconCircle(s, "secret", 0.55, 1.25, 0.55, CARD_DK);
  s.addText("Benami evasion", { x: 1.25, y: 1.33, w: 3.8, h: 0.4, fontFace: BODY, fontSize: 16, bold: true, color: GOLD, margin: 0 });
  s.addText([
    { text: "The principal files a clean, plausible return. The assets sit with frontmen — drivers, relatives — in the same household. His own books are spotless: spreadsheets and tabular ML clear him.", options: { breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "Our knowledge graph links households. Wealth hidden one hop away becomes a signal, and the graph neural network recognises the network shape itself.", options: {} },
  ], { x: 0.55, y: 1.95, w: 4.30, h: 2.7, fontFace: BODY, fontSize: 14, color: "FFFFFF", valign: "top" });
  // comparison cards
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 5.30, y: 1.30, w: 2.00, h: 2.55, rectRadius: 0.09, fill: { color: CARD_DK } });
  s.addText("7%", { x: 5.30, y: 1.55, w: 2.00, h: 1.0, fontFace: HEAD, fontSize: 54, bold: true, color: PALE, align: "center" });
  s.addText("of hidden principals caught by tabular ML", { x: 5.45, y: 2.70, w: 1.70, h: 0.9, fontFace: BODY, fontSize: 12, color: PALE, align: "center", valign: "top" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.55, y: 1.05, w: 2.10, h: 3.05, rectRadius: 0.09, fill: { color: GOLD } });
  s.addText("90%", { x: 7.55, y: 1.40, w: 2.10, h: 1.1, fontFace: HEAD, fontSize: 60, bold: true, color: GREEN_D, align: "center" });
  s.addText("caught by our graph AI — same data, same wall", { x: 7.70, y: 2.70, w: 1.80, h: 1.1, fontFace: BODY, fontSize: 12.5, bold: true, color: "41300A", align: "center", valign: "top" });
  s.addText("Measured on 2,365 hidden proxy networks inside a 50,000-person simulation. Only the graph differs.", {
    x: 0.55, y: 4.95, w: 9.0, h: 0.4, fontFace: BODY, fontSize: 12, italic: true, color: PALE,
  });
  s.addNotes("Killer demo (60s): live, open the proxy tab. This man files a clean return — a spreadsheet clears him. Expand his household: the frontmen light up holding crores. Tabular analysis catches 7 percent of these people. Our graph catches 90.");

  /* ---------------- Slide 5 — validation ---------------- */
  s = pres.addSlide();
  s.background = { color: "FFFFFF" };
  s.addText("Validated like a real deployment", {
    x: 0.5, y: 0.32, w: 9.0, h: 0.6, fontFace: HEAD, fontSize: 30, bold: true, color: GREEN_D,
  });
  const stats = [
    ["1.00", "entity-resolution precision"],
    ["0.97", "entity-resolution recall"],
    ["0.90", "detection average precision"],
    ["48", "automated tests, all green"],
  ];
  stats.forEach(([num, label], i) => {
    const x = 0.5 + i * 2.38;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.15, w: 2.18, h: 1.55, rectRadius: 0.08, fill: { color: GREEN_T }, shadow: shadow() });
    s.addText(num, { x, y: 1.28, w: 2.18, h: 0.75, fontFace: HEAD, fontSize: 40, bold: true, color: GREEN_D, align: "center" });
    s.addText(label, { x: x + 0.12, y: 2.06, w: 1.94, h: 0.55, fontFace: BODY, fontSize: 11.5, color: MUTED, align: "center", valign: "top" });
  });
  iconCircle(s, "shield", 0.55, 3.10, 0.5, GREEN_T);
  s.addText([
    { text: "The wall. ", options: { bold: true, color: GREEN_D } },
    { text: "The hidden answer key is read by exactly one file, only after detection — enforced by an automated test. Our metrics cannot be circular.", options: { color: INK } },
  ], { x: 1.22, y: 3.06, w: 8.3, h: 0.62, fontFace: BODY, fontSize: 13.5, valign: "middle", margin: 0 });
  iconCircle(s, "check", 0.55, 3.90, 0.5, GREEN_T);
  s.addText([
    { text: "Live honesty check. ", options: { bold: true, color: GREEN_D } },
    { text: "One button in the dashboard re-scores real citizens through the demo form and matches the pipeline to the decimal — judges can watch the proof run.", options: { color: INK } },
  ], { x: 1.22, y: 3.86, w: 8.3, h: 0.62, fontFace: BODY, fontSize: 13.5, valign: "middle", margin: 0 });
  s.addText("Hard cases planted and passed: same-name father and son, twins, masked CNICs, Urdu/English transliteration.", {
    x: 0.55, y: 4.95, w: 9.0, h: 0.4, fontFace: BODY, fontSize: 12, italic: true, color: MUTED,
  });
  s.addNotes("Anticipate the synthetic-data objection before judges raise it: we generated the data, so we firewalled the answer key. The detector is fully unsupervised; evaluation happens after the fact, like a real audit campaign.");

  /* ---------------- Slide 6 — market ---------------- */
  s = pres.addSlide();
  s.background = { color: "FFFFFF" };
  s.addText("Two markets, one engine", {
    x: 0.5, y: 0.32, w: 9.0, h: 0.6, fontFace: HEAD, fontSize: 30, bold: true, color: GREEN_D,
  });
  // GovTech card
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 1.10, w: 4.42, h: 2.65, rectRadius: 0.09, fill: { color: GREEN_T }, shadow: shadow() });
  iconCircle(s, "landmark", 0.75, 1.32, 0.5, GREEN_M);
  s.addText("Primary: sovereign GovTech", { x: 1.40, y: 1.38, w: 3.4, h: 0.4, fontFace: BODY, fontSize: 15, bold: true, color: GREEN_D, margin: 0 });
  s.addText([
    { text: "Direct attack on the FBR tax-gap mandate", options: { bullet: true, breakLine: true } },
    { text: "One point of tax-to-GDP is worth billions a year — pays for itself on a handful of recovered cases", options: { bullet: true, breakLine: true } },
    { text: "Read-only layer over existing databases: no new law, no disruption", options: { bullet: true } },
  ], { x: 0.80, y: 1.95, w: 3.95, h: 1.7, fontFace: BODY, fontSize: 12, color: INK, paraSpaceAfter: 6, valign: "top" });
  // AML card
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 5.10, y: 1.10, w: 4.42, h: 2.65, rectRadius: 0.09, fill: { color: GREEN_T }, shadow: shadow() });
  iconCircle(s, "bank", 5.35, 1.32, 0.5, GREEN_M);
  s.addText("Adjacent: banking KYC / AML", { x: 6.00, y: 1.38, w: 3.4, h: 0.4, fontFace: BODY, fontSize: 15, bold: true, color: GREEN_D, margin: 0 });
  s.addText([
    { text: "Benami detection is structurally identical to money-mule and shell-network detection", options: { bullet: true, breakLine: true } },
    { text: "SBP-mandated AML compliance and beneficial-ownership discovery", options: { bullet: true, breakLine: true } },
    { text: "Same engine, zero re-architecture — doubles the addressable market", options: { bullet: true } },
  ], { x: 5.40, y: 1.95, w: 3.95, h: 1.7, fontFace: BODY, fontSize: 12, color: INK, paraSpaceAfter: 6, valign: "top" });
  // growth path
  s.addText("Growth path", { x: 0.5, y: 4.00, w: 3.0, h: 0.35, fontFace: BODY, fontSize: 13, bold: true, color: GREEN_D, margin: 0 });
  const phases = [
    "Pilot: one province's excise + DISCO data",
    "Expand silos: travel, land registry, bank feeds",
    "License the same engine to banks for AML",
  ];
  phases.forEach((t, i) => {
    const x = 0.5 + i * 3.15;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 4.40, w: 2.85, h: 0.72, rectRadius: 0.08, fill: { color: i === 2 ? GOLD : "FFFFFF" }, line: { color: i === 2 ? GOLD : "C9D8D0", width: 1 } });
    s.addText(t, { x: x + 0.12, y: 4.40, w: 2.61, h: 0.72, fontFace: BODY, fontSize: 11, color: i === 2 ? "41300A" : INK, align: "center", valign: "middle", margin: 0 });
    if (i < 2) s.addText("→", { x: x + 2.85, y: 4.40, w: 0.30, h: 0.72, fontFace: BODY, fontSize: 16, bold: true, color: MUTED, align: "center", valign: "middle", margin: 0 });
  });
  s.addText("Each new silo compounds the graph's power. Blocking keeps matching near-linear — Neo4j + Spark at national scale.", {
    x: 0.5, y: 5.18, w: 9.0, h: 0.35, fontFace: BODY, fontSize: 11.5, italic: true, color: MUTED,
  });
  s.addNotes("Value (30s): the buyer is the state — even a fraction of a point of tax-to-GDP pays for this many times over. And the same engine sells to banks for AML, which de-risks the business.");

  /* ---------------- Slide 7 — close ---------------- */
  s = pres.addSlide();
  s.background = { color: GREEN_D };
  s.addText("Tabular analysis nabs the frontmen.", {
    x: 0.6, y: 1.05, w: 8.8, h: 0.65, fontFace: HEAD, fontSize: 32, bold: true, color: "FFFFFF", align: "center",
  });
  s.addText("Graph AI nabs the kingpin.", {
    x: 0.6, y: 1.70, w: 8.8, h: 0.70, fontFace: HEAD, fontSize: 36, bold: true, color: GOLD, align: "center",
  });
  const pillars = [
    ["link", "Works on data the state already holds"],
    ["clipGold", "Every flag explainable in plain PKR"],
    ["bankGold", "Same engine powers bank AML"],
  ];
  pillars.forEach(([key, label], i) => {
    const x = 0.85 + i * 2.90;
    s.addShape(pres.shapes.OVAL, { x: x + 0.95, y: 2.95, w: 0.62, h: 0.62, fill: { color: CARD_DK } });
    s.addImage({ data: ic[key], x: x + 1.11, y: 3.11, w: 0.30, h: 0.30 });
    s.addText(label, { x, y: 3.70, w: 2.55, h: 0.70, fontFace: BODY, fontSize: 13, color: "FFFFFF", align: "center", valign: "top" });
  });
  s.addText("Pilot-ready. Let us show you the live demo.", {
    x: 0.6, y: 4.75, w: 8.8, h: 0.45, fontFace: BODY, fontSize: 16, italic: true, color: PALE, align: "center",
  });
  s.addNotes("Close (20s): recite the two lines, gesture at the three pillars, invite them to the dashboard. This is deployable, not a science project.");

  await pres.writeFile({ fileName: "F:/tax-net/pitch/tax-net-pitch.pptx" });
  console.log("deck written");
}

main().catch((e) => { console.error(e); process.exit(1); });
