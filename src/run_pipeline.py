"""
One-command pipeline runner.

    python src/run_pipeline.py

Runs Stage 1-4c in order, then prints the evaluation scorecard.
The Streamlit dashboard (Stage 5) is launched separately:
    streamlit run app.py
"""

import generate_data
import entity_resolution
import build_graph
import scoring
import gnn_detector
import ensemble
import rule_floors
import audit_report
import evaluate


def main():
    print("\n[1/6] Generating synthetic civic data ...")
    generate_data.main()
    print("\n[2/6] Entity resolution ...")
    entity_resolution.resolve()
    print("\n[3/6] Building knowledge graph + features ...")
    build_graph.build()
    print("\n[4/6] Isolation-Forest deviation scoring + audit trails ...")
    scoring.score()
    print("\n[4b] GraphSAGE relational detector (skips cleanly if torch absent) ...")
    gnn_detector.run()
    print("\n[4c] Ensembling Isolation Forest + GNN into one production score ...")
    ensemble.combine()
    print("\n[4d] Rule-based lifestyle floors (explainable cross-check) ...")
    rule_floors.compute_floors()
    print("\n[4e] Generating bilingual audit notices for top flagged ...")
    audit_report.make_audits(top_n=25)
    print("\n[6/6] Evaluation (reads ground truth — scorecard only) ...")
    evaluate.evaluate_entity_resolution()
    evaluate.evaluate_detection()
    print("\nDone. Launch the dashboard with:  streamlit run app.py")


if __name__ == "__main__":
    main()
