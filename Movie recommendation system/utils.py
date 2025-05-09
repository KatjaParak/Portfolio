import pandas as pd 
from functools import lru_cache
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
# https://pypi.org/project/imdby/
import imdb
import spacy

nlp = spacy.load("en_core_web_sm")
ia = imdb.IMDb()

@lru_cache
def read_file():    
    movies = pd.read_csv("Data/movies.csv")
    tags = pd.read_csv("Data/tags.csv")
    links = pd.read_csv("Data/links.csv")

    return movies, tags, links

@lru_cache
def extract_features(): 
    movies, tags, links  = read_file()
    
    filtered_movies = movies.copy()
    filtered_tags = tags.copy()
    filtered_links = links.copy()

    filtered_movies[["movie_title", "year"]] = filtered_movies["title"].str.rsplit(n=1, expand=True)
    filtered_movies["genres"], filtered_movies["movie_title"] = filtered_movies["genres"].str.replace('|', ','), filtered_movies["movie_title"].str.replace(',', '') 

    filtered_tags.dropna(inplace=True)
    filtered_tags.drop(["userId","timestamp"],axis=1,inplace=True)

    filtered_links.drop("tmdbId", axis=1, inplace=True)

    sia = SentimentIntensityAnalyzer()
    compound_score = []
    for tag in filtered_tags["tag"]:
        scores = sia.polarity_scores(tag)
        compound = scores["compound"]
        compound_score.append(compound)

    tags_with_scores = filtered_tags.copy()
    tags_with_scores["scores"] = compound_score

    years = filtered_movies[["year", "movieId"]]

    movies_with_tags = filtered_movies.merge(tags_with_scores, on="movieId").reset_index(drop=True)
    movies_with_links = movies_with_tags.merge(filtered_links, on="movieId").reset_index(drop=True)
    movies_with_links = movies_with_links[movies_with_links["scores"] >= 0.015]    
    movies_with_links.drop(["scores"], axis=1,inplace=True)

    movies_df = movies_with_links.groupby(["movieId"]).agg({"tag": lambda x: ','.join(x.unique()),
                                                            "genres": lambda x: ','.join(x.unique()),
                                                            "movie_title": "first",
                                                            "imdbId": "first"})
    movies_df["genre_and_tag"] = movies_df["genres"] + ',' + movies_df["tag"]
    movies_df.drop(["genres", "tag"],axis=1,inplace=True)
    movies_df = movies_df.merge(years,on="movieId").reset_index(drop=True)

    return movies_df


def normalize_titles(movies_df):
    new_titles = []
    for title in movies_df["movie_title"]:
        new_title = re.sub(r'^(.*) (The|A|An)$', r'\2 \1', title)
        new_titles.append(new_title)
    
    movies_df["title"] = new_titles
    movies_df.drop("movie_title", axis=1, inplace=True)

    return movies_df


def vectorizer(movies_df):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies_df["genre_and_tag"])

    return tfidf_matrix


def cos_similarity(tfidf_matrix):
    similarity_score = cosine_similarity(tfidf_matrix)
    
    return similarity_score


def similarity_df(similarity_score, movies_df):
    sim_df = pd.DataFrame(similarity_score, index=movies_df["title"], columns=movies_df["title"])

    return sim_df

def remove_stopwords_title(sim_df):
    filtered_titles = []

    for title in sim_df.index:
        doc = nlp(title)
        filtered_title = ' '.join([token.text.lower() for token in doc if not token.is_stop])
        filtered_titles.append(filtered_title)
    return filtered_titles

# adapted from an example found on GeeksforGeeks
def get_recommendations(filtered_movie_name,sim_df,similarity_score, movies_df, filtered_titles):
    try:
        index = filtered_titles.index(filtered_movie_name)
    except ValueError:
        return None, None
    similar_movies = sorted(list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True)[1:6]
    recommendations = []
    posters = []
    covers = []

    for index, _ in similar_movies:
        temp_df = movies_df[movies_df["title"] == sim_df.index[index]]
        recommendations.append((temp_df["title"].values[0], temp_df["year"].values[0]))
        posters.append(temp_df["imdbId"].values[0])

    for poster in posters: 
        series = ia.get_movie(poster)
        covers.append(series.data['cover url'])  

    return recommendations, covers