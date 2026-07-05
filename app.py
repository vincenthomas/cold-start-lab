"""Streamlit demo: watch each strategy react as a cold user warms up.

Pick a held-out user, slide their revealed history from 0 to 5 items, and
compare what every strategy recommends side by side. The point of the demo
is the transition — how quickly each strategy escapes the popularity list.
"""
import pandas as pd
import streamlit as st

from src.data import cold_start_split, load_ml1m
from src.models import ALL_STRATEGIES

st.set_page_config(page_title="Cold-Start Lab", layout="wide")


@st.cache_resource(show_spinner="Training strategies (one-time, ~1 min)...")
def setup():
    ratings, movies, users = load_ml1m()
    train, profiles, test_users = cold_start_split(ratings)
    strategies = []
    for cls in ALL_STRATEGIES:
        s = cls()
        s.fit(train, movies, users)
        strategies.append(s)
    titles = movies.set_index("item").title
    return strategies, profiles, test_users, users.set_index("user"), titles


strategies, profiles, test_users, user_rows, titles = setup()

st.title("Cold-Start Lab")
st.caption(
    "Five strategies, one held-out user, a history slider. "
    "Every user below was excluded from training."
)

col1, col2 = st.columns([1, 1])
user = col1.selectbox("Held-out user", test_users[:200])
k = col2.slider("Interactions revealed to the strategies", 0, 5, 0)

items = profiles[user]
profile, targets = items[:k], set(items[k:])
row = user_rows.loc[user]
st.markdown(
    f"**User {user}** — {row.gender}, age bracket {row.age} · "
    f"profile: {len(profile)} items revealed, {len(targets)} future positives hidden"
)
if k > 0:
    st.markdown("Revealed history: " + " · ".join(f"_{titles.get(i, i)}_" for i in profile))

cols = st.columns(len(strategies))
for col, strat in zip(cols, strategies):
    with col:
        st.subheader(strat.name)
        if k < strat.min_profile:
            st.caption("needs ≥1 interaction — this is the cold-start blind spot")
            continue
        recs = strat.recommend(profile, row, 10)
        hits = 0
        for i in recs:
            hit = i in targets
            hits += hit
            st.markdown(("✅ " if hit else "· ") + str(titles.get(i, i)))
        st.caption(f"{hits}/10 are in this user's hidden future positives")
