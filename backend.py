import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split, GridSearchCV

# Set up the working directory
WORK_DIR = os.getcwd()

# Load datasets
def load_data():
    ratings = pd.read_csv(os.path.join(WORK_DIR, "ratings_small.csv"))
    movies = pd.read_csv(os.path.join(WORK_DIR, "movies_metadata.csv"), low_memory=False)
    keywords = pd.read_csv(os.path.join(WORK_DIR, "keywords.csv"))
    return ratings, movies, keywords

# Data preprocessing
def preprocess_data(ratings, movies, keywords):
    # Handle missing values and convert data types
    movies = movies.dropna(subset=['title', 'id'])
    movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
    movies = movies.drop_duplicates(subset=['id', 'title'])

    # Handle genres
    if movies['genres'].dtype == 'object' and not movies['genres'].str.startswith('[').any():
        movies['genres'] = movies['genres'].str.split(', ')
    else:
        import ast
        movies['genres'] = movies['genres'].fillna('[]').apply(
            lambda x: [i['name'] for i in ast.literal_eval(x)] if isinstance(x, str) else []
        )

    # Process keywords
    keywords['keywords'] = keywords['keywords'].apply(
        lambda x: x.replace(',', ' ') if isinstance(x, str) else ''
    )

    # Drop duplicates in ratings
    ratings = ratings.drop_duplicates()

    return ratings, movies, keywords

# TF-IDF vectorization for keywords
def compute_tfidf(keywords):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(keywords['keywords'])
    tfidf_scores = tfidf_matrix.mean(axis=0).tolist()[0]
    keywords_df = pd.DataFrame({'keyword': tfidf.get_feature_names_out(), 'score': tfidf_scores})
    return keywords_df

# Train collaborative filtering model
def train_model(ratings):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

    # Hyperparameter tuning
    param_grid = {
        'n_factors': [50, 100],
        'lr_all': [0.005, 0.01],
        'reg_all': [0.02, 0.1]
    }
    grid_search = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
    grid_search.fit(data)
    best_params = grid_search.best_params['rmse']

    # Train the final model
    svd = SVD(n_factors=best_params['n_factors'], 
              lr_all=best_params['lr_all'], 
              reg_all=best_params['reg_all'])
    svd.fit(trainset)
    return svd, trainset, testset

# Generate recommendations
def get_recommendations(user_id, svd_model, ratings, movies, n_recommendations=10):
    user_rated_movies = ratings[ratings['userId'] == user_id]['movieId'].unique()
    unseen_movies = movies[~movies['id'].isin(user_rated_movies)]

    predictions = []
    for movie_id in unseen_movies['id']:
        predictions.append((movie_id, svd_model.predict(user_id, movie_id).est))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_movies = unseen_movies[unseen_movies['id'].isin([p[0] for p in predictions[:n_recommendations]])]
    return top_movies[['title']].reset_index(drop=True)

# Visualization functions
def plot_genres(movies):
    all_genres = movies['genres'].explode()
    genre_counts = all_genres.value_counts()
    plt.figure(figsize=(12, 6))
    genre_counts.head(10).plot(kind='bar', color='teal')
    plt.title('Top 10 Genres in the Dataset')
    plt.xlabel('Genre')
    plt.ylabel('Frequency')
    plt.show()

def plot_ratings_distribution(ratings):
    sns.countplot(data=ratings, x='rating', palette='coolwarm')
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.show()

# Main function to run the pipeline
def main():
    # Load datasets
    ratings, movies, keywords = load_data()

    # Preprocess data
    ratings, movies, keywords = preprocess_data(ratings, movies, keywords)

    # Compute TF-IDF
    keywords_df = compute_tfidf(keywords)

    # Train collaborative filtering model
    svd_model, trainset, testset = train_model(ratings)

    # Evaluate the model
    predictions = svd_model.test(testset)
    print("Model Evaluation:")
    print(f"RMSE: {accuracy.rmse(predictions):.4f}")
    print(f"MAE: {accuracy.mae(predictions):.4f}")

    # Generate recommendations for a user
    user_id = int(input("Enter a User ID for recommendations: "))
    recommendations = get_recommendations(user_id, svd_model, ratings, movies)
    print("\nTop Movie Recommendations:")
    print(recommendations)

    # Visualizations
    plot_genres(movies)
    plot_ratings_distribution(ratings)

if __name__ == "__main__":
    main()
