import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os
from dotenv import load_dotenv
import re

# =========================
# MUST BE FIRST
# =========================
st.set_page_config(page_title="Movie Recommender", layout="wide")

# =========================
# LOAD ENV
# =========================
load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY")

# =========================
# STYLE (SMALL UI UPGRADE)
# =========================
st.markdown("""
<style>
img {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    ratings = ratings.head(100000)
    return movies, ratings

# =========================
# PREPROCESS
# =========================
@st.cache_data
def preprocess(movies, ratings):
    user_item = ratings.pivot_table(
        index='userId',
        columns='movieId',
        values='rating'
    ).fillna(0)

    item_sim = cosine_similarity(user_item.T)

    movies['genres'] = movies['genres'].fillna("").apply(lambda x: x.split('|'))
    genre_matrix = movies['genres'].str.join('|').str.get_dummies()
    content_sim = cosine_similarity(genre_matrix)

    min_dim = min(item_sim.shape[0], content_sim.shape[0])

    hybrid = (
        0.7 * item_sim[:min_dim, :min_dim] +
        0.3 * content_sim[:min_dim, :min_dim]
    )

    return movies, hybrid

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
# RECOMMEND
# =========================
def recommend(movie_title, movies, hybrid, top_n=5):
    idx = movies[movies['title'] == movie_title].index

    if len(idx) == 0:
        return []

    idx = idx[0]

    if idx >= hybrid.shape[0]:
        return []

    scores = list(enumerate(hybrid[idx]))
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
# LOAD
# =========================
with st.spinner("⚡ Loading model..."):
    movies, ratings = load_data()
    movies, hybrid = preprocess(movies, ratings)

# =========================
# UI HEADER
# =========================
st.markdown("# 🎬 Hybrid Movie Recommendation System")

# =========================
# INPUT
# =========================
movie_list = movies['title'].dropna().unique()
selected_movie = st.selectbox("🔍 Search Movie", movie_list)

top_n = st.slider("Number of recommendations", 3, 10, 5)

# =========================
# BUTTON
# =========================
if st.button("🔥 Get Recommendations"):

    with st.spinner("Finding best matches..."):
        recs = recommend(selected_movie, movies, hybrid, top_n)

    if not recs:
        st.error("Movie not found")

    else:
        # =========================
        # TOP RECOMMENDATION (CENTERED FIX)
        # =========================
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

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")

        # =========================
        # RESPONSIVE GRID
        # =========================
        st.markdown("## 🎯 More Like This")

        num_cols = 4  # adjust if needed

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

# =========================
# FOOTER
# =========================
st.markdown("---")
