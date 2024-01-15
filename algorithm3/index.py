import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../archive rotten tomatoes/rotten_tomatoes_movies.csv')
df.head()
df.columns
df.describe(include='all')
df['genres'] = df['genres'].apply(lambda x: [] if pd.isna(x) else [genre.strip() for genre in x.split(',')])
df_encoded = df['genres'].apply(lambda x: pd.Series([1] * len(x), index=x)).fillna(0).astype(int, errors='ignore')  # One-Hot Encoding
df = pd.concat([df, df_encoded], axis=1)
df = pd.get_dummies(df, columns=['content_rating'], prefix='content_rating')    # One-Hot Encoding

content_rating_columns = df.filter(like='content_rating')
for col in content_rating_columns:
    df[col] = df[col].astype(int)

def clean_tomatometer_status(status):
    if status == 'Rotten':
        return 0
    elif status == 'Fresh':
        return 1
    elif status == 'Certified-Fresh':
        return 2
    return status
df.loc[:, 'tomatometer_status'] = df['tomatometer_status'].apply(lambda x: clean_tomatometer_status(x))
df.tomatometer_status.value_counts()

df = df.dropna(subset=['tomatometer_rating', 'audience_rating', 'audience_status']) # drop rows containing NaN
df.loc[:, 'audience_status'] = df['audience_status'].apply(lambda x: 1 if x == 'Upright' else 0 if x == 'Spilled' else x )
df.audience_status.value_counts()

used_features = [
    'tomatometer_status', 'tomatometer_rating', 'audience_status', 'audience_rating',
    'Action & Adventure', 'Comedy', 'Drama', 'Science Fiction & Fantasy',
    'Romance', 'Classics', 'Kids & Family', 'Mystery & Suspense', 'Western',
    'Art House & International', 'Horror', 'Faith & Spirituality',
    'Animation', 'Documentary', 'Special Interest',
    'Musical & Performing Arts', 'Sports & Fitness', 'Television',
    'Cult Movies', 'Anime & Manga', 'Gay & Lesbian',
    'content_rating_G', 'content_rating_NC17', 'content_rating_NR', 'content_rating_PG', 'content_rating_PG', 'content_rating_R']

df[ ['rotten_tomatoes_link']+used_features ]

df[used_features].isna().sum().sum()
# print(df[used_features].to_numpy().shape)
# print(df[used_features].to_numpy())

# df[ ['rotten_tomatoes_link']+used_features ].to_csv('/content/movies.csv',index=False)

# Load CSV data
df = pd.read_csv('../data/movies.csv', nrows = 2000)

# Extract relevant columns for similarity computation
features = df[
    ['tomatometer_rating', 'audience_rating', 'Action & Adventure', 'Comedy', 'Drama', 'Science Fiction & Fantasy',
               'Romance', 'Classics', 'Kids & Family', 'Mystery & Suspense', 'Western',
               'Art House & International', 'Horror', 'Faith & Spirituality',
               'Animation', 'Documentary', 'Special Interest',
               'Musical & Performing Arts', 'Sports & Fitness', 'Television',
               'Cult Movies', 'Anime & Manga', 'Gay & Lesbian',
               'content_rating_G', 'content_rating_NC17', 'content_rating_NR', 'content_rating_PG', 'content_rating_R']]

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features.fillna(0))  # Fill NaN values with 0 for simplicity

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(features_scaled, features_scaled)

# Convert similarity matrix to DataFrame for better interpretation
movie_titles = df['rotten_tomatoes_link']  # Assuming you have a 'movie_title' column in your DataFrame
cosine_sim_df = pd.DataFrame(cosine_sim, index=movie_titles, columns=movie_titles)

# Set a similarity threshold (adjust as needed)
threshold = 0.5

# Create an adjacency matrix based on the threshold
adjacency_matrix = np.where(cosine_sim > threshold, 1, 0)

# Convert the matrix to a DataFrame for better interpretation
adjacency_matrix_df = pd.DataFrame(adjacency_matrix, index=movie_titles, columns=movie_titles)

def adjacency_matrix_to_adjacency_list(adjacency_matrix):
    adjacency_list = {}

    for i in range(len(adjacency_matrix)):
        neighbors = [j for j in range(len(adjacency_matrix[i])) if adjacency_matrix[i][j] == 1]
        adjacency_list[i] = neighbors

    return adjacency_list

adjacency_list = adjacency_matrix_to_adjacency_list(adjacency_matrix)
# for node, neighbors in adjacency_list.items():
#     print(f"{node}: {neighbors}")

G = nx.from_dict_of_lists(adjacency_list)

def find_k_cliques(adjacency_matrix):
    k_cliques = [clique for clique in nx.find_cliques(G)]
    return k_cliques

k_cliques = find_k_cliques(adjacency_matrix)

def remove_duplicates(cliques):
    unique_cliques = set(tuple(sorted(clique)) for clique in cliques)
    return [list(clique) for clique in unique_cliques]

cliques = remove_duplicates(k_cliques)
# for i in cliques:
#   print (i)


# Replace 42 with the line number you want to read (zero-based index)
line_number = 4001

# Read the specific line
new_movie_data = pd.read_csv('../data/movies.csv', skiprows=line_number, nrows=1)

# Initialize variables to keep track of similarity counts for each cluster
cluster_similarity_counts = {tuple(cluster): 0 for cluster in cliques}

# Compute similarity to existing movies in each cluster
for cluster in cliques:

    # Extract features of movies in the current cluster
    cluster_features = features_scaled[cluster]

    # Compute cosine similarity between the new movie and movies in the cluster
    similarity_scores = cosine_similarity(features, cluster_features)

    # Count the number of similar movies (similarity score above a threshold)
    num_similar_movies = np.sum(similarity_scores > 0.5)

    # Update the count for the current cluster
    cluster_similarity_counts[tuple(cluster)] = num_similar_movies

# Choose the cluster with the highest count as the most similar cluster
most_similar_cluster = max(cluster_similarity_counts, key=cluster_similarity_counts.get)

print(f"The new moxvie belongs to the most similar cluster: {most_similar_cluster}")

reviews_df = pd.read_csv('../data/reviews.csv')
selected_reviews = reviews_df[['Source', 'Target','top_critic']]
movie_reviewers_dict = {}

# Iterate through each row in the reviews DataFrame
for _, row in reviews_df.iterrows():
    movie_id = row['Target']
    reviewer = row['Source']
    top_critic_value = row['top_critic']

    # If the movie_id is not in the dictionary, add it with an empty list
    if movie_id not in movie_reviewers_dict:
        movie_reviewers_dict[movie_id] = []

    # Append the reviewer to the list of reviewers for the corresponding movie
    movie_reviewers_dict[movie_id].append(reviewer)

# Convert the dictionary to a list of tuples
movie_reviewers_list = [(movie_name, reviewers) for movie_name, reviewers in movie_reviewers_dict.items()]
print(movie_reviewers_list[0])

movies_titles_cluster = list(movie_titles.iloc[list(most_similar_cluster)])

# Create an empty dictionary to store reviewer frequencies
reviewer_top_critic_sum = {}

# Iterate through each movie in the cluster
for movie_title in movies_titles_cluster:
    # Find the tuple corresponding to the current movie
    movie_tuple = next((t for t in movie_reviewers_list if t[0] == movie_title), None)

    if movie_tuple:
        # Get the list of reviewers for the current movie
        reviewers_for_movie = movie_tuple[1]

        # Count the frequency of each reviewer
        for reviewer in reviewers_for_movie:
             # If the reviewer is not in the dictionary, add it with a default sum of 0
            if reviewer not in reviewer_top_critic_sum:
                reviewer_top_critic_sum[reviewer] = 0

            # Add the 'top_critic' value to the sum for the corresponding reviewer
            reviewer_top_critic_sum[reviewer] += top_critic_value

reviewer_top_critic_list = list(reviewer_top_critic_sum.items())

sorted_reviewer_top_critic = sorted(reviewer_top_critic_list, key=lambda x: x[1], reverse=True)

# Print the resulting dictionary with reviewer frequencies
for i, (reviewer, frequency) in enumerate(sorted_reviewer_top_critic[:5], 1):
    print(f"{i}. Reviewer: {reviewer},")
