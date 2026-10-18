import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

#Load the Dataset
df = pd.read_csv('upload_a573aba5-b7e9-4e1f-a29a-3273c13ef095.csv')
print('--- Initial Dataset Info ---')
print(df.info())

#Data Pre-processing Operations
print('\n--- Pre-processing ---')
print('Null values before dropping:', df.isnull().sum().sum())

# Drop missing values and duplicates
df = df.dropna().drop_duplicates()
print('Dataset shape after cleaning:', df.shape)

# Select numerical features for analysis and clustering
audio_features = [
    'danceability',
    'energy',
    'key',
    'loudness',
    'mode',
    'speechiness',
    'acousticness',
    'instrumentalness',
    'liveness',
    'valence',
    'tempo',
]

#Data Analysis & Visualizations
 Plot 1: Distribution of Playlist Genres
plt.figure(figsize=(10, 5))
sns.countplot(
    data=df, x='playlist_genre', palette='Set2', hue='playlist_genre', legend=False
)
plt.title('Distribution of Songs Across Playlist Genres')
plt.xlabel('Playlist Genre')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('genre_distribution.png')
plt.show()

 #Correlation Matrix of Audio Features
plt.figure(figsize=(10, 8))
corr_matrix = df[audio_features].corr()
sns.heatmap(
    corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5
)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.show()

 #Clustering using K-Means
# Standardize the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[audio_features])

# Apply K-Means clustering (let's use k=5 clusters)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(scaled_data)

 Plot 2: Clusters based on Danceability and Energy
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x='danceability',
    y='energy',
    hue='cluster',
    palette='viridis',
    alpha=0.6,
)
plt.title('K-Means Clusters: Danceability vs Energy')
plt.xlabel('Danceability')
plt.ylabel('Energy')
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig('clusters_dance_energy.png')
plt.show()

# 6. Recommendation System Foundation (Cosine Similarity)
# We can recommend songs based on feature similarity to a given track
similarity_matrix = cosine_similarity(scaled_data)


def recommend_songs(track_name, num_recommendations=5):
  try:
    idx = df[df['track_name'].str.lower() == track_name.lower()].index[0]
    distances = sorted(
        list(enumerate(similarity_matrix[idx])),
        reverse=True,
        key=lambda x: x[1],
    )
    print(f"\nRecommendations because you liked '{track_name}':")
    for i in distances[1 : num_recommendations + 1]:
      rec_track = df.iloc[i[0]]['track_name']
      rec_artist = df.iloc[i[0]]['track_artist']
      print(f'- {rec_track} by {rec_artist}')
  except IndexError:
    print('Song not found in the dataset!')


# Test recommendation system with the first track in dataset
sample_song = df.iloc[0]['track_name']
recommend_songs(sample_song)
