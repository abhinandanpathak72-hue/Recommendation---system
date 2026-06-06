import streamlit as st
import pickle
from sklearn.metrics.pairwise import linear_kernel
from difflib import get_close_matches

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.main {
    background-color: #0f1117;
}

.header {
    text-align:center;
    padding:20px;
}

.header h1{
    color:#E50914;
    font-size:60px;
    margin-bottom:0;
}

.header p{
    color:white;
    font-size:20px;
}

.movie-card{
    background:#1e1e1e;
    padding:20px;
    border-radius:15px;
    text-align:center;
    color:white;
    margin-bottom:15px;
    transition:0.3s;
    box-shadow:0px 4px 10px rgba(0,0,0,0.5);
}

.movie-card:hover{
    transform:scale(1.05);
    box-shadow:0px 6px 20px rgba(229,9,20,0.6);
}

.stButton > button{
    background-color:#E50914;
    color:white;
    border:none;
    border-radius:10px;
    height:50px;
    width:100%;
    font-size:18px;
    font-weight:bold;
}

.stButton > button:hover{
    background-color:#ff2d37;
}

div[data-baseweb="select"]{
    color:black;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD FILES
# ==================================================

@st.cache_resource
def load_data():

    df = pickle.load(open("df.pkl", "rb"))
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
    tfidf_matrix = pickle.load(open("tfidf_matrix.pkl", "rb"))
    indices = pickle.load(open("indices.pkl", "rb"))

    return df, tfidf, tfidf_matrix, indices


df, tfidf, tfidf_matrix, indices = load_data()

# ==================================================
# RECOMMENDATION FUNCTION
# ==================================================

def recommend(movie_title, n=10):

    try:

        idx = indices[movie_title]

        similarity_scores = linear_kernel(
            tfidf_matrix[idx],
            tfidf_matrix
        ).flatten()

        movie_indices = similarity_scores.argsort()[::-1][1:n+1]

        recommendations = df['title'].iloc[movie_indices]

        return recommendations.tolist()

    except:
        return []


# ==================================================
# FUZZY SEARCH FUNCTION
# ==================================================

def find_closest_movie(movie_name):

    all_movies = df['title'].tolist()

    matches = get_close_matches(
        movie_name,
        all_movies,
        n=1,
        cutoff=0.5
    )

    return matches[0] if matches else None


# ==================================================
# HEADER
# ==================================================

st.markdown("""
<div class='header'>
<h1>🎬 Movie Recommender</h1>
<p>Discover movies similar to your favorites</p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# SEARCH SECTION
# ==================================================

movie_list = sorted(df['title'].unique())

selected_movie = st.selectbox(
    "Search Movie",
    movie_list,
    index=None,
    placeholder="Type movie name..."
)

# ==================================================
# BUTTON
# ==================================================

if st.button("🚀 Recommend Movies"):

    if selected_movie is None:

        st.warning("Please select a movie.")

    else:

        with st.spinner("Finding similar movies..."):

            recommendations = recommend(selected_movie)

        if len(recommendations) == 0:

            st.error("No recommendations found.")

        else:

            st.success(
                f"Movies similar to '{selected_movie}'"
            )

            cols = st.columns(5)

            for i, movie in enumerate(recommendations):

                with cols[i % 5]:

                    st.markdown(
                        f"""
                        <div class='movie-card'>
                            <h4>{movie}</h4>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

