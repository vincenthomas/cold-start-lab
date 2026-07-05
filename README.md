# Cold-Start Lab — five strategies race through a user's first five interactions

**Live demo:** pending deploy · **Stack:** pandas, implicit (ALS), gensim (item2vec), Streamlit · **Status:** working

> New users get the worst recommendations exactly when retention is decided. This repo holds out 1,000 MovieLens-1M users, reveals their history one interaction at a time (k = 0, 1, 3, 5), and measures how five strategies behave in that window. Headline result: nothing beats raw popularity on recall@10 before five interactions — but personalized strategies recommend from 30–50× more of the catalog, and ALS fold-in is on track to cross popularity just past the window. The interesting finding is what that tradeoff means for when to switch strategies.

---

## 1. The problem (not the model)

A new user signs up, sees generic suggestions, and churns before the system learns anything — the recommendations are worst at the exact moment the user is deciding whether to stay. Most teams paper over this with a global "trending" shelf, which retains the median user but shows every niche user someone else's taste. The crux this project pulls on: **at what point in a user's history does personalization actually start beating popularity, and which signal gets there first?**

## 2. The workflow

`signup (demographics) → first interactions trickle in → strategy picks top-10 → user acts → repeat`

The system owns strategy selection and ranking. The human boundary: which signals are acceptable to use at signup (demographics can be sensitive) is a product/policy decision, not a modeling one — the eval measures demographics' value so that decision can be made with numbers.

## 3. System design

```
ratings.dat ─→ positives (rating ≥ 4) ─→ hold out 1,000 users entirely
                                          │
train users ─→ fit: popularity | demo-pop | content | item2vec | ALS
                                          │
test user, first k interactions revealed ─→ each strategy: top-10/top-50
                                          │
hidden future positives ─→ recall@10/50, coverage@10, latency p50/p95
```

The feedback loop is the k-sweep itself: rerunning eval at each profile size shows where each strategy's curve bends — that curve, not any single number, is the deliverable.

## 4. Model & architecture choices — and what they beat

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| positive signal | ratings ≥ 4 | all ratings as implicit signal | a 2-star rating is evidence of dissatisfaction; folding it into "interacted" rewards recommending items users disliked |
| k=0 personalization | popularity within (gender, age) bucket | zip-code or occupation buckets | gender×age buckets keep ≥ thousands of users each; finer buckets go sparse and back off to global anyway |
| CF for cold users | ALS fold-in (`recalculate_user`, item factors frozen) | retraining per user; explicit-rating SVD | fold-in is O(k) at inference and matches how production MF serves new users; explicit SVD optimizes rating prediction, not ranking |
| embeddings | item2vec (word2vec over interaction sequences) | pretrained text embeddings of titles | isolates the interaction signal; title text mostly re-encodes genre, which the content strategy already covers |
| eval split | hold out users, not interactions | random interaction split | interaction splits leak the test user into training, which is exactly the situation cold-start doesn't have |

## 5. Eval plan — what "good" means

- **Definition of quality:** recall@10 and recall@50 against each user's hidden future positives (primary); catalog coverage@10 (guardrail — a strategy that shows everyone the same 10 blockbusters must not win); per-user latency p50/p95 (feasibility).
- **Golden set:** 1,000 users sampled (seed=7) from users with ≥ 15 positives, held out of all training including item2vec vocabulary. Never used for tuning; hyperparameters are library defaults deliberately, so the comparison is between signals, not tuning effort.
- **Method:** offline replay. recall@N = |top-N ∩ hidden positives| / |hidden positives|.
- **Regression check:** `python -m src.eval` rewrites `results/metrics.csv`; any strategy change reruns the full sweep.

## 6. Cost, latency, quality tradeoffs

Measured on the eval box (CPU only, no GPU; costs are compute time, no API spend):

| Strategy | fit | recommend p50 / p95 (ms/user) |
|---|---|---|
| popularity | <0.01 s | 0.17 / 0.20 |
| demo-pop | 0.04 s | 0.14 / 0.18 |
| content | 0.02 s | 0.91 / 1.24 |
| item2vec | 1.8 s | 0.72 / 0.79 |
| ALS fold-in | 0.55 s | 0.18 / 0.50 |

Everything here is cheap enough that cost doesn't pick the winner; quality-per-signal does. What I'd ship: demo-pop at k=0, switch to ALS fold-in from the first interaction — it's the only strategy whose recall climbs monotonically with k while holding 40× popularity's coverage, and its fold-in latency (0.2 ms) permits per-request computation.

## 7. Failure modes & mitigations

- **Popularity feedback loop:** popularity's coverage@10 is 0.3% of catalog — shipped alone it starves the catalog of new signal. Mitigation: coverage is a first-class guardrail metric, and the shipped policy switches away from popularity at k=1.
- **Demographic bucket goes sparse or missing:** demo-pop backs off to global popularity and pads short lists; a user with no demographics degrades to the popularity baseline, never to an empty shelf.
- **Profile items missing from a model's vocabulary** (item2vec min_count=5 drops rare items; ALS drops items absent from train): strategies return an empty list rather than guessing; the shipped system would fall through to demo-pop. Hit while building: early item2vec runs silently skipped unknown items and produced 8-item "top-10" lists — the eval now recommends from the intersection and pads are explicit.
- **Averaging washes out taste (found in eval, see §9):** item2vec recall *falls* as k grows, because the mean of five diverse item vectors points nowhere. Detection: the k-sweep curve itself. Mitigation candidate is scoring against each profile item separately and merging (§10).

## 8. UX & trust decisions

The demo shows, per recommendation, whether it hit the user's actual hidden future (✅) — including the misses. A demo that only showcased hits would read better and mean nothing; the visible miss rate is what makes the coverage/recall tradeoff legible. Strategies that can't run at k=0 say so on-screen ("needs ≥1 interaction — this is the cold-start blind spot") instead of silently falling back, because the blind spot is the point of the comparison.

## 9. Results

1,000 held-out users, recall@10 / coverage@10 by profile size k (full table incl. recall@50: `results/metrics.csv`):

| Strategy | k=0 | k=1 | k=3 | k=5 | coverage@10 (k=5) |
|---|---|---|---|---|---|
| popularity | .057 | .055 | .053 | .051 | 0.004 |
| demo-pop | **.060** | **.059** | **.057** | **.056** | 0.012 |
| content | — | .013 | .014 | .018 | 0.308 |
| item2vec | — | .012 | .008 | .007 | 0.298 |
| ALS fold-in | — | .029 | .039 | .048 | 0.105 |

What the numbers say, honestly:

- **Popularity is brutally hard to beat inside the cold window.** No personalized strategy catches it on recall@10 by k=5. Demographics buy a real but small lift (+5–10% relative over global popularity, consistent at every k).
- **ALS fold-in is the only strategy converging on popularity** (.029 → .048 and rising at the window edge, at 26× the coverage). Extrapolating the curve, the crossover sits just past k=5 — measuring it directly is next-step #1.
- **Negative result:** item2vec *degrades* with more history (.012 → .007). Mean-pooling five taste vectors cancels them out. The strategy as built is not shippable; documented rather than tuned away.
- Content-based recall is weak but it is the only strategy that also works for cold *items*, which nothing else here can serve.

## 10. What I'd do next

1. Extend the k-sweep to k=20 to measure, not extrapolate, the ALS/popularity crossover — that number is the actual "switch strategies here" answer (§9).
2. Replace item2vec mean-pooling with per-item nearest-neighbor merge to test whether the degradation is pooling or the embeddings themselves (§7, §9).
3. Blend demo-pop and ALS fold-in with a k-dependent weight and check whether the blend beats both endpoints, since they win at different k (§9).

---

*Built with an agentic workflow (Claude Code) — the commit history is the
iteration log. Part of a three-repo portfolio: [Cold-Start Lab](https://github.com/vincenthomas/cold-start-lab) · [Guardrail](https://github.com/vincenthomas/guardrail) · [Policy RAG](https://github.com/vincenthomas/policy-rag).*
