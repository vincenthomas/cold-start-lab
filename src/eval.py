"""Cold-start evaluation: recall@k and coverage across profile sizes.

Run:  python -m src.eval
Writes results/metrics.csv and prints a markdown table.

Metrics:
- recall@N  = |top-N recs ∩ held-out positives| / |held-out positives|
- coverage@10 = unique items in all users' top-10 / catalog size
  (a strategy that recommends the same 10 blockbusters to everyone scores
  high recall and terrible coverage; both numbers matter)
- latency: per-user recommend() wall time, p50/p95
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import cold_start_split, load_ml1m
from .models import ALL_STRATEGIES

PROFILE_SIZES = [0, 1, 3, 5]
TOP_N = [10, 50]


def evaluate(data_dir: str = "data", n_test_users: int = 1000, seed: int = 7):
    ratings, movies, users = load_ml1m(data_dir)
    train, profiles, test_users = cold_start_split(ratings, n_test_users, seed=seed)
    user_rows = users.set_index("user")
    catalog_size = movies.item.nunique()

    rows = []
    for cls in ALL_STRATEGIES:
        strat = cls()
        t0 = time.perf_counter()
        strat.fit(train, movies, users)
        fit_s = time.perf_counter() - t0

        for k in PROFILE_SIZES:
            if k < strat.min_profile:
                continue
            recalls = {n: [] for n in TOP_N}
            rec_sets = set()
            lat = []
            for u in test_users:
                items = profiles[u]
                profile, targets = items[:k], set(items[k:])
                if not targets:
                    continue
                t1 = time.perf_counter()
                recs = strat.recommend(profile, user_rows.loc[u], max(TOP_N))
                lat.append(time.perf_counter() - t1)
                for n in TOP_N:
                    hits = len(set(recs[:n]) & targets)
                    recalls[n].append(hits / len(targets))
                rec_sets.update(recs[:10])
            rows.append({
                "strategy": strat.name, "profile_size": k,
                "recall@10": np.mean(recalls[10]), "recall@50": np.mean(recalls[50]),
                "coverage@10": len(rec_sets) / catalog_size,
                "fit_seconds": round(fit_s, 2),
                "latency_p50_ms": round(1000 * float(np.percentile(lat, 50)), 2),
                "latency_p95_ms": round(1000 * float(np.percentile(lat, 95)), 2),
                "n_users": len(recalls[10]),
            })
            print(f"{strat.name:12s} k={k}  r@10={rows[-1]['recall@10']:.4f} "
                  f"r@50={rows[-1]['recall@50']:.4f} cov={rows[-1]['coverage@10']:.3f}")

    df = pd.DataFrame(rows)
    out = Path("results")
    out.mkdir(exist_ok=True)
    df.to_csv(out / "metrics.csv", index=False)
    print("\n" + df.to_markdown(index=False, floatfmt=".4f"))
    return df


if __name__ == "__main__":
    evaluate()
