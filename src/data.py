"""Data loading and the cold-start evaluation protocol for MovieLens-1M.

Protocol: hold out N test users entirely. For each test user, their first-k
positive interactions (by timestamp) form the "profile" a strategy is allowed
to see; every later positive is an eval target. Sweeping k in {0, 1, 3, 5}
traces each strategy's behavior across the cold-start window, which is the
comparison this repo exists to make.

Design decision: ratings >= 4 count as positive signal; 1-3 star ratings are
discarded rather than treated as negatives. Alternative considered: use all
ratings as implicit signal. Rejected because a 2-star rating is evidence of
dissatisfaction, and folding it into "interacted" rewards recommending items
users disliked.
"""
from pathlib import Path

import numpy as np
import pandas as pd

POSITIVE_THRESHOLD = 4


def load_ml1m(data_dir: str = "data"):
    d = Path(data_dir)
    ratings = pd.read_csv(
        d / "ratings.dat", sep="::", engine="python", encoding="latin-1",
        names=["user", "item", "rating", "ts"],
    )
    movies = pd.read_csv(
        d / "movies.dat", sep="::", engine="python", encoding="latin-1",
        names=["item", "title", "genres"],
    )
    users = pd.read_csv(
        d / "users.dat", sep="::", engine="python", encoding="latin-1",
        names=["user", "gender", "age", "occupation", "zip"],
    )
    return ratings, movies, users


def cold_start_split(ratings: pd.DataFrame, n_test_users: int = 1000,
                     min_positives: int = 15, seed: int = 7):
    """Split into train interactions and per-test-user chronological positives.

    Test users need >= min_positives so that even at k=5 there are >= 10
    targets left to evaluate against. Returns (train_df, profiles, test_users)
    where profiles[u] is u's positive items in timestamp order.
    """
    pos = ratings[ratings.rating >= POSITIVE_THRESHOLD].sort_values("ts")
    counts = pos.groupby("user").size()
    eligible = counts[counts >= min_positives].index.to_numpy()
    rng = np.random.default_rng(seed)
    test_users = set(rng.choice(eligible, size=n_test_users, replace=False).tolist())
    train = pos[~pos.user.isin(test_users)]
    test = pos[pos.user.isin(test_users)]
    profiles = {u: g.item.to_numpy() for u, g in test.groupby("user")}
    return train, profiles, sorted(test_users)
