import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import requests
import re

# =========================
# PAGE CONFIG
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
# LOAD DATA (LIMITED)
# =========================
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv").head(3000)
    ratings = pd.read_csv("ratings.csv").head(5000)
    return movies, ratings

# =========================
# PREPROCESS
# =========================
@st.cache_data
def preprocess(movies):
    movies['genres'] = movies['genres'].fillna("").apply(lambda x: x.split('|'))
    return movies

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
# SIMPLE SIMILARITY
# =========================
def simple_similarity(g1, g2):
    set1, set2 = set(g1), set(g2)
    return len(set1 & set2) / max(len(set1 | set2), 1)

# =========================
# RECOMMEND FUNCTION
# =========================
def recommend(movie_title, movies, top_n=5):
    target = movies[movies['title'] == movie_title]

    if target.empty:
        return []

    target_genres = target.iloc[0]['genres']

    scores = []

    for i, row in movies.iterrows():
        score = simple_similarity(target_genres, row['genres'])
        scores.append((i, score))

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
# METRICS
# =========================
def calculate_rmse(ratings):
    if len(ratings) == 0:
        return 0

    sample = ratings.sample(min(1000, len(ratings)))
    actuals = sample['rating']
    preds = np.full(len(actuals), actuals.mean())

    return np.sqrt(mean_squared_error(actuals, preds))

# =========================
# LOAD
# =========================
with st.spinner("⚡ Loading model..."):
    movies, ratings = load_data()
    movies = preprocess(movies)

# =========================
# HEADER
# =========================
st.title("🎬 Movie Recommendation System")
st.info("Lightweight content-based recommender optimized for real-time deployment")

# =========================
# METRICS
# =========================
st.subheader("📊 Model Performance")

rmse = calculate_rmse(ratings)

col1, col2 = st.columns(2)
col1.metric("Baseline RMSE", f"{rmse:.3f}")
col2.metric("Model Type", "Content-Based (Optimized)")

st.divider()

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

    recs = recommend(selected_movie, movies, top_n)

    if not recs:
        st.error("Movie not found")

    else:
        # TOP
        st.subheader("⭐ Top Recommendation")

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

        st.divider()

        # GRID
        st.subheader("🎯 More Like This")

        cols = st.columns(4)

        for i, rec in enumerate(recs[1:]):
            with cols[i % 4]:
                poster = fetch_poster(rec['title'])

                if poster:
                    st.image(poster, width=180)
                else:
                    st.image("https://via.placeholder.com/300x450?text=No+Image")

                st.markdown(f"**{rec['title']}**")
                st.progress(rec['score'])
                st.caption(f"{rec['genres']}")

# =========================
# FOOTER
# =========================
st.divider()
st.caption("Built with Python • Streamlit • Lightweight ML")