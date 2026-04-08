import streamlit as st
import pandas as pd
import requests
import re

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Movie Recommender", layout="wide")

# =========================
# API KEY (SAFE)
# =========================
try:
    API_KEY = st.secrets["TMDB_API_KEY"]
except:
    API_KEY = None

# =========================
# LOAD DATA (VERY SMALL)
# =========================
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv").head(1500)  # VERY SMALL
    return movies

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
def fetch_poster(title):
    if not API_KEY:
        return None
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={clean_title(title)}"
        data = requests.get(url).json()
        if data["results"]:
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        pass
    return None

# =========================
# SIMPLE RECOMMENDER
# =========================
def recommend(movie_title, movies, top_n=5):
    target = movies[movies['title'] == movie_title]

    if target.empty:
        return []

    target_genres = set(target.iloc[0]['genres'])

    scores = []

    for i, row in movies.iterrows():
        genres = set(row['genres'])
        score = len(target_genres & genres)  # simple overlap
        scores.append((i, score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    results = []
    for i, score in scores:
        results.append({
            "title": movies.iloc[i]['title'],
            "score": score,
            "genres": ", ".join(movies.iloc[i]['genres'])
        })

    return results

# =========================
# LOAD
# =========================
movies = preprocess(load_data())

# =========================
# UI
# =========================
st.title("🎬 Movie Recommender")
st.caption("Fast & lightweight recommendation system (deployment-safe)")

movie_list = movies['title'].dropna().unique()
selected_movie = st.selectbox("Select a movie", movie_list)

top_n = st.slider("Number of recommendations", 3, 10, 5)

# =========================
# BUTTON
# =========================
if st.button("Get Recommendations"):

    recs = recommend(selected_movie, movies, top_n)

    if not recs:
        st.error("No recommendations found")

    else:
        st.subheader("Top Recommendation")

        top = recs[0]

        poster = fetch_poster(top['title'])
        if poster:
            st.image(poster, width=250)

        st.markdown(f"### {top['title']}")
        st.caption(f"Genres: {top['genres']}")

        st.divider()

        st.subheader("More Like This")

        cols = st.columns(3)

        for i, rec in enumerate(recs[1:]):
            with cols[i % 3]:
                poster = fetch_poster(rec['title'])
                if poster:
                    st.image(poster, width=150)

                st.markdown(f"**{rec['title']}**")
                st.caption(rec['genres'])