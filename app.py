import streamlit as st
import joblib
import difflib
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- LOAD MODELS ----------------
import os

@st.cache_resource
def load_data(filename):
    if not os.path.exists(filename):
        # Check if split parts exist
        part_num = 0
        parts = []
        while os.path.exists(f"{filename}.part{part_num}"):
            parts.append(f"{filename}.part{part_num}")
            part_num += 1
            
        if parts:
            print(f"Reconstructing {filename} from {len(parts)} parts... This may take a moment.")
            with open(filename, 'wb') as output_file:
                for part in parts:
                    with open(part, 'rb') as part_file:
                        output_file.write(part_file.read())
            print(f"Reconstructed {filename}")
        else:
            st.error(f"Error: {filename} not found. Please ensure all model files are present.")
            st.stop()
            return None
            
    return joblib.load(filename)

movies_tags = load_data("movies_tags.pkl")
vectors = load_data("vectors.pkl")
similarity = load_data("similarity.pkl")
cv = load_data("vectorizer.pkl")

# ---------------- CONSTANTS ----------------
TMDB_API_KEY = st.secrets["TMDB_API_KEY"]

GENRE_KEYWORDS = {
    "fantasy": [
        "magic", "myth", "mythical", "dragon", "kingdom",
        "sword", "legend", "ancient", "epic", "quest"
    ],
    "science fiction": [
        "future", "technology", "space", "alien", "robot"
    ],
    "romance": [
        "love", "relationship", "emotion", "heart"
    ],
    "war": [
        "battle", "army", "soldier"
    ]
}

MOOD_KEYWORDS = {
    "Dark & Gritty": ["dark", "crime", "violence", "noir", "gritty"],
    "Light & Funny": ["funny", "comedy", "laugh", "cheerful", "happy"],
    "Mind-Bending": ["psychological", "complex", "mystery", "puzzle", "twist"],
    "Adrenaline": ["action", "fast", "chase", "explosion", "thrill"],
    "Heartwarming": ["family", "love", "friendship", "wholesome", "inspiring"]
}

# ---------------- HELPERS ----------------
def normalize_text(text):
    text = text.lower()
    text = ''.join(ch for ch in text if ch.isalnum() or ch == ' ')
    return text

def get_explanation(source_vec, target_vec):
    # Ensure dense arrays
    if hasattr(source_vec, "toarray"): source_vec = source_vec.toarray()
    if hasattr(target_vec, "toarray"): target_vec = target_vec.toarray()
    
    source_vec = np.ravel(source_vec)
    target_vec = np.ravel(target_vec)
    
    # Calculate contribution of each term to the similarity (dot product)
    interaction = source_vec * target_vec
    
    # Get top 3 terms
    feature_names = cv.get_feature_names_out()
    top_indices = interaction.argsort()[-3:][::-1]
    
    terms = [feature_names[i] for i in top_indices if interaction[i] > 0]
    return terms

@st.cache_data
def fetch_movie_details(movie_title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
    try:
        data = requests.get(url).json()
        if data['results']:
            movie = data['results'][0]
            poster_path = movie.get('poster_path')
            poster = "https://image.tmdb.org/t/p/w500" + poster_path if poster_path else "https://via.placeholder.com/150x225?text=No+Poster"
            return {
                "poster": poster,
                "title": movie.get('title', movie_title),
                "rating": movie.get('vote_average', 'N/A'),
                "date": movie.get('release_date', 'N/A'),
                "overview": movie.get('overview', '')
            }
    except:
        pass
    return None

def find_movie(movie_name):
    movie_name = normalize_text(movie_name)
    titles = movies_tags['title'].str.lower().tolist()
    matches = difflib.get_close_matches(movie_name, titles, n=1, cutoff=0.4)
    if matches:
        return movies_tags[movies_tags['title'].str.lower() == matches[0]].iloc[0]['title']
    return None

def recommend_existing(movie):
    idx = movies_tags[movies_tags['title'] == movie].index[0]
    scores = similarity[idx]

    recs = sorted(
        list(enumerate(scores)),
        key=lambda x: x[1],
        reverse=True
    )[1:6]
    
    final_recs = []
    source_vec = vectors[idx]
    
    for i, score in recs:
        explanation = get_explanation(source_vec, vectors[i])
        final_recs.append((movies_tags.iloc[i].title, round(score, 3), explanation))

    return final_recs

def recommend_cold(description, genres, mood=None):
    description = normalize_text(description)
    genres_clean = normalize_text(genres)

    # expand genre keywords (Iterate keys to handle "science fiction")
    expanded = []
    for category, keywords in GENRE_KEYWORDS.items():
        if category in genres_clean:
            expanded.extend(keywords)

    # expand mood keywords
    if mood and mood in MOOD_KEYWORDS:
        expanded.extend(MOOD_KEYWORDS[mood])

    expanded_text = " ".join(expanded)

    # 🔥 weighted composition
    # Weight Priorities: Description (1x) + Genres (2x) + Hidden Keywords (2x)
    tags = f"""
    {description}
    {genres_clean} {genres_clean}
    {expanded_text} {expanded_text}
    """

    tags = normalize_text(tags)

    new_vec = cv.transform([tags]).toarray()
    scores = cosine_similarity(new_vec, vectors)[0]

    recs = sorted(
        list(enumerate(scores)),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    final_recs = []
    
    for i, score in recs:
        # vectors[i] might be sparse, get_explanation handles it
        explanation = get_explanation(new_vec, vectors[i])
        final_recs.append((movies_tags.iloc[i].title, round(score, 3), explanation))

    return final_recs

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.title("🎬 Movie Recommendation System")
st.markdown("##### Discover your next favorite movie using AI 🍿")

st.sidebar.title("🧭 Navigation")
mode = st.sidebar.radio(
    "Choose recommendation mode:",
    ["🍿 Existing Movie", "✨ New Movie (Cold Start)"]
)

st.sidebar.divider()
st.sidebar.info("💡 **Tip:** Use the **Cold Start** mode to describe an idea you have in mind and let the AI find the closest matches!")

st.divider()

if mode == "🍿 Existing Movie":
    st.subheader("Search for a movie")
    
    # Examples
    st.write("Try these examples:")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Inception", use_container_width=True): st.session_state.existing_query = "Inception"
    if c2.button("Toy Story", use_container_width=True): st.session_state.existing_query = "Toy Story"
    if c3.button("Interstellar", use_container_width=True): st.session_state.existing_query = "Interstellar"
    if c4.button("Titanic", use_container_width=True): st.session_state.existing_query = "Titanic"

    st.markdown("<br>", unsafe_allow_html=True)
    movie_name = st.text_input("Enter movie name:", key="existing_query", placeholder="e.g. The Matrix")

    if st.button("Recommend", type="primary"):
        with st.spinner("Finding best matches..."):
            movie = find_movie(movie_name)
            if movie is None:
                st.error("Movie not found in dataset. Please try another one.")
            else:
                st.success(f"Selected: **{movie}**")
                st.subheader("Because you watched this:")
                
                results = recommend_existing(movie)
            
            for title, score, explanation in results:
                details = fetch_movie_details(title)
                with st.container(border=True):
                    c1, c2 = st.columns([1, 4])
                    with c1:
                        if details:
                            st.image(details["poster"], use_container_width=True)
                    with c2:
                        st.subheader(f"{title} ({details['date'][:4] if details else 'N/A'})")
                        if details:
                            st.caption(f"⭐ {details['rating']}/10")
                            st.write(f"_{details['overview'][:120]}..._")
                        
                        if explanation:
                            st.write(f"💡 **Why:** {', '.join(explanation)}")

else:
    st.subheader("Describe your dream movie")
    
    # Examples
    st.write("Quick start examples:")
    ex_c1, ex_c2 = st.columns(2)
    
    # Initialize session state if not present to avoid KeyErrors
    if "desc_input" not in st.session_state: st.session_state.desc_input = ""
    if "genre_input" not in st.session_state: st.session_state.genre_input = ""
    if "mood_input" not in st.session_state: st.session_state.mood_input = "None"

    if ex_c1.button("🚀 Cyberpunk Detective", use_container_width=True):
        st.session_state.desc_input = "A detective hunting androids in a neon city future with flying cars."
        st.session_state.genre_input = "science fiction mystery"
        st.session_state.mood_input = "Dark & Gritty"
        st.rerun()
        
    if ex_c2.button("🏰 Epic Fantasy Quest", use_container_width=True):
        st.session_state.desc_input = "A group of heroes goes on a journey to save the kingdom from a dragon."
        st.session_state.genre_input = "fantasy adventure"
        st.session_state.mood_input = "Adrenaline"
        st.rerun()

    col1, col2 = st.columns([2, 1])

    with col1:
        description = st.text_area(
            "Movie description:",
            placeholder="E.g., A robot travels back in time to save a human leader...",
            key="desc_input"
        )

    with col2:
        genres = st.text_input(
            "Genres:",
            placeholder="sci-fi action",
            key="genre_input"
        )
        mood = st.selectbox(
            "Vibe / Mood (Optional):",
            ["None"] + list(MOOD_KEYWORDS.keys()),
            key="mood_input"
        )

    if st.button("Generate Recommendations", type="primary"):
        if description.strip() == "":
            st.error("Please enter a description.")
        else:
            with st.spinner("Analyzing your vibe and generating recommendations..."):
                mood_val = mood if mood != "None" else None
                
                st.subheader("Here are your recommendations:")
                results = recommend_cold(description, genres, mood_val)
            
            # Display in a grid
            for title, score, explanation in results:
                details = fetch_movie_details(title)
                with st.container(border=True):
                    c1, c2 = st.columns([1, 4])
                    with c1:
                        if details:
                            st.image(details["poster"], use_container_width=True)
                    with c2:
                        st.subheader(f"{title} ({details['date'][:4] if details else 'N/A'})")
                        if details:
                            st.caption(f"⭐ {details['rating']}/10")
                            st.write(f"_{details['overview'][:120]}..._")
                            
                        if explanation:
                            st.write(f"💡 **Matches:** {', '.join(explanation)}")
