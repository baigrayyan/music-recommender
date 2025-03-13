from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Flask app
app = Flask(__name__)

# Load your dataset
try:
    df = pd.read_csv('clustered_df.csv')
except FileNotFoundError:
    print("Error: 'clustered_df.csv' file not found.")
    exit(1)

numerical_features = [
    "valence", "danceability", "energy", "tempo",
    "acousticness", "liveness", "speechiness", "instrumentalness"
]

def recommend_songs(song_name, df, num_recommendations=5):
    # Check if the song exists in the DataFrame
    if song_name not in df["name"].values:
        raise ValueError("Song not found in the dataset.")

    # Get the cluster of the input song
    song_cluster = df[df["name"] == song_name]["Cluster"].values[0]
    same_cluster_songs = df[df["Cluster"] == song_cluster]
    
    # Check if the song is in the same cluster
    if same_cluster_songs.empty:
        raise ValueError("No songs found in the same cluster.")

    song_index = same_cluster_songs[same_cluster_songs["name"] == song_name].index[0]
    cluster_features = same_cluster_songs[numerical_features]
    similarity = cosine_similarity(cluster_features, cluster_features)
    
    # Get similar songs
    similar_songs = np.argsort(similarity[song_index])[-(num_recommendations + 1):-1][::-1]
    recommendations = same_cluster_songs.iloc[similar_songs][["name", "year", "artists"]]
    return recommendations

# Route for the home page
@app.route("/")
def index():
    return render_template('index.html')

# Route for recommendations
@app.route("/recommend", methods=["POST"])
def recommend():
    recommendations = []
    if request.method == "POST":
        song_name = request.form.get("song_name")
        try:
            recommendations = recommend_songs(song_name, df).to_dict(orient="records")
        except ValueError as e:
            recommendations = [{"name": "Error", "artists": str(e), "year": ""}]
        except Exception as e:
            recommendations = [{"name": "Error", "artists": "An unexpected error occurred", "year": ""}]
    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)