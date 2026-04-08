import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import requests
import re

# =========================
# PAGE CONFIG (MUST BE FIRST)
# =========================
st.set_page_config(page_title="Movie Recommender", layout="wide")

# =========================
# SAFE API KEY
# =========================
try:
    API_KEY = st.secrets["TMDB_API_KEY"]
except:
    API_KEY = None

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    ratings = ratings.head(20000)  # lightweight
    return movies, ratings

# =========================
# LIGHTWEIGHT PREPROCESS
# =========================
@st.cache_data
def preprocess(movies):
    movies['genres'] = movies['genres'].fillna("").apply(lambda x: x.split('|'))
    genre_matrix = movies['genres'].str.join('|').str.get_dummies()
    return movies, genre_matrix

# =========================
# CLEAN TITLE
# =========================
def clean_title(title):
    title = re.sub(r"\(\d{4}\)", "", title)
    title = title.replace(", The", "")
    return title.strip()

# =========================
# FETCH POSTER
# =========================
def fetch_poster(movie_title):
    if not API_KEY:
        return None

    try:
        movie_title = clean_title(movie_title)
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}"
        data = requests.get(url).json()

        if data.get("results"):
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"

        return None
    except:
        return None

# =========================
# RECOMMEND (ON-DEMAND SIMILARITY)
# =========================
def recommend(movie_title, movies, genre_matrix, top_n=5):
    idx = movies[movies['title'] == movie_title].index

    if len(idx) == 0:
        return []

    idx = idx[0]

    target = genre_matrix.iloc[idx].values.reshape(1, -1)
    sim_scores = cosine_similarity(target, genre_matrix)[0]

    scores = list(enumerate(sim_scores))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    results = []
    for i, score in scores:
        results.append({
            "title": movies.iloc[i]['title'],
            "score": round(score, 3),
            "genres": ", ".join(movies.iloc[i]['genres'])
        })

    return results

# =========================
# SAFE RMSE (BASELINE)
# =========================
def calculate_rmse(ratings):
    if len(ratings) == 0:
        return 0

    sample_size = min(1000, len(ratings))
    sample = ratings.sample(sample_size)

    actuals = sample['rating']
    preds = np.full(len(actuals), actuals.mean())

    return np.sqrt(mean_squared_error(actuals, preds))

# =========================
# LOAD
# =========================
with st.spinner("⚡ Loading model..."):
    movies, ratings = load_data()
    movies, genre_matrix = preprocess(movies)

# =========================
# HEADER
# =========================
st.markdown("# 🎬 Hybrid Movie Recommendation System")
st.info("🔍 Lightweight ML recommender using content-based similarity (deployment optimized)")

# =========================
# METRICS
# =========================
st.markdown("## 📊 Model Performance")

rmse = calculate_rmse(ratings)

col1, col2 = st.columns(2)

with col1:
    st.metric("Baseline RMSE", f"{rmse:.3f}")

with col2:
    st.metric("Model Type", "Content-Based (Optimized)")

st.markdown("---")

# =========================
# INPUT
# =========================
movie_list = movies['title'].dropna().unique()
selected_movie = st.selectbox("🔍 Search Movie", movie_list)

top_n = st.slider("Recommendations", 3, 10, 5)

# =========================
# BUTTON
# =========================
if st.button("🔥 Get Recommendations"):

    with st.spinner("🔎 Finding movies you'll love..."):
        recs = recommend(selected_movie, movies, genre_matrix, top_n)

    if not recs:
        st.error("Movie not found")

    else:
        # TOP
        st.markdown("## ⭐ Top Recommendation")

        top = recs[0]

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            poster = fetch_poster(top['title'])

            if poster:
                st.image(poster, width=300)
            else:
                st.image("https://via.placeholder.com/300x450?text=No+Image")

            st.markdown(f"### {top['title']}")
            st.progress(top['score'])
            st.caption(f"⭐ Score: {top['score']}")
            st.caption(f"🎭 {top['genres']}")

        st.markdown("---")

        # GRID
        st.markdown("## 🎯 More Like This")

        num_cols = 4
        remaining = recs[1:]

        rows = [remaining[i:i + num_cols] for i in range(0, len(remaining), num_cols)]

        for row in rows:
            cols = st.columns(len(row))

            for col, rec in zip(cols, row):
                with col:
                    poster = fetch_poster(rec['title'])

                    if poster:
                        st.image(poster, width=180)
                    else:
                        st.image("https://via.placeholder.com/300x450?text=No+Image")

                    st.markdown(f"**{rec['title']}**")
                    st.progress(rec['score'])
                    st.caption(f"Because of: {rec['genres']}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Built with Python • Streamlit • Optimized ML for Deployment")