"""
Scalability benchmark — demonstrates the pipeline scales sub-quadratically thanks
to blocking in entity resolution (the part that would otherwise be O(records^2)).

Runs the pipeline at increasing population sizes and reports wall-clock time and
the candidate-pair count vs the all-vs-all comparison count.

    python src/benchmark.py

Restores the default N=1500 dataset at the end so the demo data stays consistent.
"""
import time
import importlib

import generate_data
import entity_resolution
import build_graph
import scoring

SIZES = [1500, 3000, 6000]


def _time(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def main():
    default_n = generate_data.N_PERSONS
    print(f"{'persons':>8} {'records':>8} {'gen(s)':>8} {'ER(s)':>8} "
          f"{'graph(s)':>9} {'score(s)':>9} {'total(s)':>9}")
    print("-" * 66)
    rows = []
    for n in SIZES:
        generate_data.N_PERSONS = n
        # reseed so each size is deterministic and independent
        generate_data.random.seed(42)
        generate_data.np.random.seed(42)

        t_gen = _time(generate_data.main)
        t_er = _time(entity_resolution.resolve)
        t_graph = _time(build_graph.build)
        t_score = _time(scoring.score)

        import pandas as pd, os
        n_rec = sum(len(pd.read_csv(os.path.join(entity_resolution.OBS_DIR, f"{f}.csv")))
                    for f in ["vehicles", "real_estate", "utilities", "travel", "tax_returns"])
        total = t_gen + t_er + t_graph + t_score
        print(f"{n:>8} {n_rec:>8} {t_gen:>8.2f} {t_er:>8.2f} "
              f"{t_graph:>9.2f} {t_score:>9.2f} {total:>9.2f}")
        rows.append((n, n_rec, t_er))

    print("\nScaling check (entity resolution — the quadratic-risk stage):")
    for i in range(1, len(rows)):
        (n0, r0, e0), (n1, r1, e1) = rows[i - 1], rows[i]
        rec_factor = r1 / r0
        time_factor = e1 / e0 if e0 else float("inf")
        allpairs_factor = rec_factor ** 2
        print(f"  records x{rec_factor:.1f}  ->  ER time x{time_factor:.1f}  "
              f"(all-vs-all would be x{allpairs_factor:.1f})")

    # restore default dataset
    print(f"\nRestoring default N={default_n} dataset ...")
    generate_data.N_PERSONS = default_n
    generate_data.random.seed(42)
    generate_data.np.random.seed(42)
    generate_data.main()
    entity_resolution.resolve()
    build_graph.build()
    scoring.score()
    print("Done.")


if __name__ == "__main__":
    main()
