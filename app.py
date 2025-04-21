from flask import Flask, render_template, request
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from ast import literal_eval

app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv('merged_data.csv')
df['Genres'] = df['Genres'].apply(literal_eval)
df["MovieID"] -= 1
df["Gender"] = df["Gender"].astype('category').cat.codes
df["Occupation"] = df["Occupation"].astype('category').cat.codes

# Age grouping (same as used in training)
def get_age_group(age):
    if age < 10:
        return 0
    elif age < 15:
        return 1
    elif age < 20:
        return 2
    elif age < 25:
        return 3
    elif age < 35:
        return 4
    elif age < 50:
        return 5
    else:
        return 6


df["AgeGroup"] = df["Age"].apply(get_age_group)

# Feature dimensions
num_movies = df['MovieID'].max() + 1
num_genders = df['Gender'].max() + 1
num_occupations = df['Occupation'].max() + 1
num_age_groups = 7
all_genres = sorted(set(g for sublist in df['Genres'] for g in sublist))
genre_to_idx = {genre: i for i, genre in enumerate(all_genres)}
num_genres = len(all_genres)

# Movie name lookup
movie_names = {row['MovieID']: row['Title'] for _, row in df.iterrows()}


# Updated model
class FederatedRecommender(nn.Module):
    def __init__(self, num_movies, num_genders, num_occupations, num_genres, num_age_groups, embedding_dim=10):
        super(FederatedRecommender, self).__init__()
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.gender_embedding = nn.Embedding(num_genders, embedding_dim)
        self.occupation_embedding = nn.Embedding(num_occupations, embedding_dim)
        self.age_embedding = nn.Embedding(num_age_groups, embedding_dim)
        self.genre_projection = nn.Linear(num_genres, embedding_dim)

        self.fc1 = nn.Linear(embedding_dim * 5, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, age_group, movie, gender, occupation, genres):
        movie_emb = self.movie_embedding(movie)
        gender_emb = self.gender_embedding(gender)
        occupation_emb = self.occupation_embedding(occupation)
        age_emb = self.age_embedding(age_group)
        genre_emb = self.genre_projection(genres.float())
        
        x = torch.cat([age_emb, movie_emb, gender_emb, occupation_emb, genre_emb], dim=-1)
        x = torch.relu(self.fc1(x))
        rating = self.fc2(x)
        return rating.squeeze()


# Load trained model
model = FederatedRecommender(
    num_movies=num_movies,
    num_genders=num_genders,
    num_occupations=num_occupations,
    num_genres=num_genres,
    num_age_groups=num_age_groups
)
model.load_state_dict(torch.load("federated_model.pth", map_location=torch.device('cpu')))
model.eval()


def prepare_input(age, movie_id, gender_str, occupation_str, genres_list):
    age_group = get_age_group(age)
    age_tensor = torch.tensor([age_group], dtype=torch.long)
    movie_tensor = torch.tensor([movie_id], dtype=torch.long)
    gender_code = pd.Series([gender_str]).astype('category').cat.codes[0]
    occupation_code = pd.Series([occupation_str]).astype('category').cat.codes[0]
    gender_tensor = torch.tensor([gender_code], dtype=torch.long)
    occupation_tensor = torch.tensor([occupation_code], dtype=torch.long)
    
    genre_vec = torch.zeros(num_genres)
    for genre in genres_list:
        if genre in genre_to_idx:
            genre_vec[genre_to_idx[genre]] = 1
    genre_tensor = genre_vec.unsqueeze(0)
    
    return age_tensor, movie_tensor, gender_tensor, occupation_tensor, genre_tensor


def recommend_movies(age, gender, occupation, genres, top_n=5):
    all_movie_ids = range(num_movies)
    predictions = []

    for movie_id in all_movie_ids:
        inputs = prepare_input(age, movie_id, gender, occupation, genres)
        with torch.no_grad():
            rating = model(*inputs).item()
            rating = np.clip(rating, 1.0, 5.0)
            predictions.append((movie_id, rating))

    top_movies = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
    return [movie_names.get(mid, f"Movie {mid}") for mid, _ in top_movies]


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        age = int(request.form['age'])
        gender = request.form['gender']
        occupation = request.form['occupation']
        genres = request.form['genres'].split(',')
        recommendations = recommend_movies(age, gender, occupation, genres)
        return render_template('index.html', recommendations=recommendations)
    return render_template('index.html', recommendations=None)


if __name__ == '__main__':
    app.run(debug=True)
