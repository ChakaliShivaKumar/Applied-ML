from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from ast import literal_eval

# Initialize Flask app
app = Flask(__name__)

# Assuming the federated model and other functions are already defined elsewhere
# Example model loading
# Load the federated model for prediction
df = pd.read_csv('merged_data.csv')
# Convert string representation of list to actual list for Genres column
df['Genres'] = df['Genres'].apply(literal_eval)
# Step 1: Zero-Index all categorical features
df["UserID"] = df["UserID"] - 1  # Zero-index UserID
df["MovieID"] = df["MovieID"] - 1  # Zero-index MovieID
df["Gender"] = df["Gender"].astype('category').cat.codes  # Zero-index Gender
df["Occupation"] = df["Occupation"].astype('category').cat.codes  # Zero-index Occupation

num_users = df['UserID'].max() + 1
num_movies = df['MovieID'].max() + 1
num_genders = df['Gender'].max() + 1
num_occupations = df['Occupation'].max() + 1
all_genres = set()
for genres in df['Genres']:
    all_genres.update(genres)
all_genres = sorted(all_genres)
num_genres = len(all_genres)
genre_to_idx = {genre: i for i, genre in enumerate(all_genres)}


# Example Federated Recommender Model using PyTorch
class FederatedRecommender(nn.Module):
    def __init__(self, num_users, num_movies, num_genders, num_occupations, num_genres, embedding_dim=10):
        super(FederatedRecommender, self).__init__()
        
        # Embeddings for each feature
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.gender_embedding = nn.Embedding(num_genders, embedding_dim)
        self.occupation_embedding = nn.Embedding(num_occupations, embedding_dim)
        
        # Genre embeddings - we'll use a linear layer
        self.genre_projection = nn.Linear(num_genres, embedding_dim)
        
        # Fully connected layers for rating prediction
        self.fc1 = nn.Linear(embedding_dim * 5, 128)  # 5 features: user, movie, gender, occupation, genres
        self.fc2 = nn.Linear(128, 1)

    def forward(self, user, movie, gender, occupation, genres):
        # Embedding lookup
        user_embedded = self.user_embedding(user)
        movie_embedded = self.movie_embedding(movie)
        gender_embedded = self.gender_embedding(gender)
        occupation_embedded = self.occupation_embedding(occupation)
        
        # Process genres - project binary flags to embedding space
        genre_embedded = self.genre_projection(genres.float())
        
        # Concatenate all embeddings
        all_embeddings = torch.cat([
            user_embedded, 
            movie_embedded, 
            gender_embedded, 
            occupation_embedded,
            genre_embedded
        ], dim=-1)
        
        # Pass through fully connected layers
        x = torch.relu(self.fc1(all_embeddings))
        rating = self.fc2(x)
        return rating.squeeze()



federated_model = FederatedRecommender(
    num_users=num_users, 
    num_movies=num_movies, 
    num_genders=num_genders, 
    num_occupations=num_occupations, 
    num_genres=num_genres
)

movie_names = {movie_id: movie_name for movie_id, movie_name in zip(df['MovieID'], df['MovieName'])}

federated_model.load_state_dict(torch.load("federated_model.pth"))
federated_model.eval()

# Function to recommend movies using the federated model
def recommend_movies_federated(user_id, gender, occupation, genres, num_recommendations=5):
    all_movie_ids = range(num_movies)  # All possible movie IDs
    movie_ratings = []

    # Loop over all movie IDs to get predictions
    for movie_id in all_movie_ids:
        # Prepare the input for the model (ensure correct format for input)
        user_tensor, movie_tensor, gender_tensor, occupation_tensor, genres_tensor = prepare_input(
            user_id, movie_id, gender, occupation, genres
        )
        
        # Predict the rating for the movie using the federated model
        with torch.no_grad():
            predicted_rating = federated_model(user_tensor, movie_tensor, gender_tensor, occupation_tensor, genres_tensor)
            predicted_rating = torch.clamp(predicted_rating, 1.0, 5.0)  # Ensure it's between 1 and 5
            movie_ratings.append((movie_id, predicted_rating.item()))

    # Sort movies by predicted rating (highest to lowest)
    movie_ratings.sort(key=lambda x: x[1], reverse=True)
    
    # Get the top N recommendations
    top_recommendations = movie_ratings[:num_recommendations]
    
    # Create a list of movie names and predicted ratings to display
    recommendations = []
    for movie_id, rating in top_recommendations:
        # Fetch the movie name from a movie_names dictionary
        movie_name = movie_names.get(movie_id, f"Unknown Movie {movie_id}")  # If no name found, fallback to the movie ID
        recommendations.append(f"{movie_name}")
    
    return recommendations

def prepare_input(user_id, movie_id, gender_str, occupation_str, genres_list):
    user = torch.tensor([user_id], dtype=torch.long)
    movie = torch.tensor([movie_id], dtype=torch.long)
    
    gender_code = pd.Series([gender_str]).astype('category').cat.codes[0]
    occupation_code = pd.Series([occupation_str]).astype('category').cat.codes[0]
    
    gender = torch.tensor([gender_code], dtype=torch.long)
    occupation = torch.tensor([occupation_code], dtype=torch.long)
    
    
    genre_vec = torch.zeros(num_genres)
    for genre in genres_list:
        if genre in genre_to_idx:
            genre_vec[genre_to_idx[genre]] = 1
    genres = genre_vec.unsqueeze(0)
    
    return user, movie, gender, occupation, genres



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user inputs from the form
        user_id = int(request.form['user_id'])
        gender = request.form['gender']
        occupation = request.form['occupation']
        genres = request.form['genres'].split(",")  # Genres are comma separated

        # Get movie recommendations based on user input
        recommendations = recommend_movies_federated(user_id, gender, occupation, genres, num_recommendations=5)

        return render_template('index.html', recommendations=recommendations)

    return render_template('index.html', recommendations=None)


if __name__ == '__main__':
    app.run(debug=True)
