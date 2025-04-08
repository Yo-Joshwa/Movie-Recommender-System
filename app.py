import streamlit as st
import pickle
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

ratings = movies[['id', 'title', 'vote_average', 'vote_count']].copy()
ratings = ratings.rename(columns={'id': 'movieId'})  
ratings = ratings[ratings['vote_count'] > 50]  
ratings['userId'] = np.random.randint(1, 1000, size=len(ratings)) 
ratings['rating'] = ratings['vote_average'] / 2  

movies = movies.merge(credits, on="title")
import ast
def extract_names(obj, key, top_n=3):
    try:
        obj = ast.literal_eval(obj) 
        names = [person[key] for person in obj[:top_n]]  
        return " ".join(names)
    except:
        return ''

movies['genres'] = movies['genres'].apply(lambda x: extract_names(x, 'name'))
movies['keywords'] = movies['keywords'].apply(lambda x: extract_names(x, 'name'))
movies['cast'] = movies['cast'].apply(lambda x: extract_names(x, 'name'))
movies['crew'] = movies['crew'].apply(lambda x: extract_names(x, 'job'))

movies['director'] = movies['crew'].apply(lambda x: x if 'Director' in x else '')

movies['combined_features'] = movies['genres'] + " " + movies['keywords'] + " " + movies['cast'] + " " + movies['director']

# movies['tags'] = movies['tags'].apply(lambda x:x.lower())
# movies['title'] = movies['title'].apply(lambda x:x.lower())

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=0.2)

svd_model = SVD()
svd_model.fit(trainset)



tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])


knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
knn_model.fit(tfidf_matrix)

movie_indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def content_recommend(title, n=5):
    if title not in movie_indices:
        return []
    
    idx = movie_indices[title]

    distances, indices = knn_model.kneighbors(tfidf_matrix[idx], n_neighbors=n+1)

    recommended_movies = [movies.iloc[i]['title'] for i in indices.flatten()[1:]]
    
    return recommended_movies


def collaborative_recommend(user_id, n=5):
    movie_ids = ratings["movieId"].unique()
    
    user_rated = ratings[ratings["userId"] == user_id]["movieId"].values
    movie_ids = [m for m in movie_ids if m not in user_rated]

    predictions = [svd_model.predict(user_id, m) for m in movie_ids]
    
    predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]
    
    recommended_ids = [p.iid for p in predictions]
    recommended_movies = movies[movies["id"].isin(recommended_ids)]["title"].tolist()
    
    return recommended_movies


def hybrid_recommend(user_id, title, alpha=0.6, n=5):
    content = content_recommend(title, n=5)
    collab = collaborative_recommend(user_id, n=5)
    # print(content)
    # print(collab)
    hybrid = list(set(content + collab))
    
    if title not in hybrid:
        hybrid.append(title)  
        
    movie_s = {movie: (alpha * content.count(movie) + (1 - alpha) * collab.count(movie))
                    for movie in hybrid}

    sorted_movies = sorted(movie_s, key=movie_s.get, reverse=True)

    if title in sorted_movies:
        sorted_movies.remove(title)
        
    sorted_movies.insert(0, title)  # Place searched movie at the top
    
    recommended_movie_posters = [fetch_poster(movie) for movie in sorted_movies[:n]]
    
    return sorted_movies[:n],recommended_movie_posters

st.markdown(
    """
    <style>
        .st-emotion-cache-13ln4jf.ea3mdgi5 {
            max-width: 95% !important;  /* Adjust this value as needed */
            margin: auto !important;  /* Center the container */
        }
    </style>
    """,
    unsafe_allow_html=True
)
movies_list = pickle.load(open('movie_list.pkl', 'rb'))

movies['title'] = movies['title'].fillna("Unknown")

st.title("Movie Recommender System")

all_options = (
    movies["title"].tolist() +
    [actor.strip() for cast in movies["cast"] for actor in cast.split(",")] +
    movies["director"].tolist()
)
selected_movie = st.selectbox("🎬 Select a Movie:", all_options)



def fetch_poster(movie_title):
    """Fetch the movie poster URL from TMDB API"""
    api_key = "8265bd1679663a7ea12ac168da84d2e8"  
    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_title}"

    response = requests.get(url)
    data = response.json()

    if data.get("results") and len(data["results"]) > 0:
        poster_path = data["results"][0].get("poster_path") 
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"  

    return "https://via.placeholder.com/500x750?text=No+Image+Available" 


if st.button("Recommend"):
    st.subheader("Recommended Movies:")
    
    recommended_movies,recommended_movie_posters = hybrid_recommend(1, selected_movie, alpha=0.6, n=10)
    
    if recommended_movies:
        cols = st.columns(5)  
        
        for i in range(5):
            with cols[i]:
                st.text(recommended_movies[i])
                st.image(recommended_movie_posters[i])

        cols = st.columns(5)  
        for i in range(5, min(10, len(recommended_movies))): 
            with cols[i - 5]: 
                st.text(recommended_movies[i])
                st.image(recommended_movie_posters[i])
        
else:
    st.write("No recommendations found.")
