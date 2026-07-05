"""Four cold-start strategies behind one interface.

Each strategy implements fit(train, movies, users) and
recommend(profile, user_row, n) -> list[item_id], so the eval loop treats
them identically. `profile` is the k revealed items; `user_row` carries
demographics (the only signal available at k=0).
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class Popularity:
    """Global popularity. The k=0 floor every other strategy must beat."""

    name = "popularity"
    min_profile = 0

    def fit(self, train, movies, users):
        self.ranked = train.item.value_counts().index.to_numpy()

    def recommend(self, profile, user_row, n):
        seen = set(profile)
        return [i for i in self.ranked if i not in seen][:n]


class DemographicPopularity:
    """Popularity within (gender, age) bucket, global backoff.

    The only personalization possible at k=0: cross the demographic signal
    from registration onto interaction data from train users.
    """

    name = "demo-pop"
    min_profile = 0

    def fit(self, train, movies, users):
        merged = train.merge(users[["user", "gender", "age"]], on="user")
        self.by_bucket = {
            b: g.item.value_counts().index.to_numpy()
            for b, g in merged.groupby(["gender", "age"])
        }
        self.global_ranked = train.item.value_counts().index.to_numpy()

    def recommend(self, profile, user_row, n):
        ranked = self.by_bucket.get((user_row.gender, user_row.age), self.global_ranked)
        seen = set(profile)
        recs = [i for i in ranked if i not in seen][:n]
        if len(recs) < n:
            have = seen | set(recs)
            recs += [i for i in self.global_ranked if i not in have][: n - len(recs)]
        return recs


class ContentProfile:
    """Genre + decade item features; user vector = mean of profile items.

    Needs k >= 1. Pure side-information: works for items nobody has rated,
    which no interaction-based strategy can claim.
    """

    name = "content"
    min_profile = 1

    def fit(self, train, movies, users):
        genres = movies.genres.str.get_dummies(sep="|")
        year = movies.title.str.extract(r"\((\d{4})\)$")[0].astype(float)
        decade = pd.get_dummies((year // 10).fillna(0).astype(int), prefix="dec")
        feats = pd.concat([genres, decade], axis=1).to_numpy(dtype=np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        self.feats = feats / np.maximum(norms, 1e-9)
        self.items = movies.item.to_numpy()
        self.index = {it: idx for idx, it in enumerate(self.items)}
        # rank ties (many items share genre vectors) by popularity, not item id
        pop = train.item.value_counts()
        self.pop = np.array([pop.get(it, 0) for it in self.items], dtype=np.float32)
        self.pop /= self.pop.max()

    def recommend(self, profile, user_row, n):
        idx = [self.index[i] for i in profile if i in self.index]
        if not idx:
            return []
        uvec = self.feats[idx].mean(axis=0)
        scores = self.feats @ uvec + 1e-3 * self.pop
        seen = set(profile)
        order = np.argsort(-scores)
        return [self.items[j] for j in order if self.items[j] not in seen][:n]


class ItemVec:
    """item2vec: word2vec over chronological interaction sequences.

    Embeddings come only from train users, so a test user's profile items are
    looked up, never trained on. Needs k >= 1.
    """

    name = "item2vec"
    min_profile = 1

    def fit(self, train, movies, users):
        from gensim.models import Word2Vec

        sentences = [
            g.item.astype(str).tolist() for _, g in train.sort_values("ts").groupby("user")
        ]
        self.w2v = Word2Vec(
            sentences, vector_size=64, window=5, min_count=5, sg=1,
            workers=4, epochs=5, seed=7,
        )
        self.items = np.array([int(w) for w in self.w2v.wv.index_to_key])
        vecs = self.w2v.wv.vectors
        self.vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
        self.index = {it: idx for idx, it in enumerate(self.items)}

    def recommend(self, profile, user_row, n):
        idx = [self.index[i] for i in profile if i in self.index]
        if not idx:
            return []
        uvec = self.vecs[idx].mean(axis=0)
        scores = self.vecs @ uvec
        seen = set(profile)
        order = np.argsort(-scores)
        return [self.items[j] for j in order if self.items[j] not in seen][:n]


class ALSFoldIn:
    """Implicit-feedback ALS trained on warm users; cold users enter by fold-in.

    Item factors are fixed at inference; the test user's factor is solved
    from their k profile items (recalculate_user=True). This is the honest
    way to give matrix factorization a shot at cold users without retraining.
    Chosen over explicit-rating SVD because thresholded implicit signal is
    what production ranking systems actually see. Needs k >= 1.
    """

    name = "als-foldin"
    min_profile = 1

    def fit(self, train, movies, users):
        import implicit

        u_codes, self.u_index = pd.factorize(train.user)
        i_codes, i_index = pd.factorize(train.item)
        self.items = i_index.to_numpy()
        self.index = {it: idx for idx, it in enumerate(self.items)}
        mat = csr_matrix(
            (np.ones(len(train), dtype=np.float32), (u_codes, i_codes)),
            shape=(len(self.u_index), len(self.items)),
        )
        self.model = implicit.als.AlternatingLeastSquares(
            factors=64, regularization=0.05, iterations=15, random_state=7,
        )
        self.model.fit(mat, show_progress=False)

    def recommend(self, profile, user_row, n):
        cols = [self.index[i] for i in profile if i in self.index]
        if not cols:
            return []
        row = csr_matrix(
            (np.ones(len(cols), dtype=np.float32), ([0] * len(cols), cols)),
            shape=(1, len(self.items)),
        )
        ids, _ = self.model.recommend(
            0, row, N=n, recalculate_user=True, filter_already_liked_items=True,
        )
        return [self.items[j] for j in ids]


ALL_STRATEGIES = [Popularity, DemographicPopularity, ContentProfile, ItemVec, ALSFoldIn]
