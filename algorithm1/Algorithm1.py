#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 18:30:47 2023

@author: nikos
"""

import pandas as pd
import os
import numpy as np
import math
 


def get_user_input(question_message, min_value, max_value):
    """
    Prompts the user for input within a specified range and ensures it is a valid integer.

    Args:
        prompt (str): The prompt message to display to the user.
        min_value (int): The minimum allowed input value.
        max_value (int): The maximum allowed input value.

    Returns:
        int: The user input within the specified range.
    """
    while True:
        # Get user input and save it to a variable
        user_input = input(question_message)
        print()

        # Check if the input is a valid integer within the specified range
        try:
            user_input = int(user_input)
            if min_value <= user_input <= max_value:
                return user_input  # Return the input if it is valid
            else:
                print(f"Invalid input. Please enter a number between {min_value} and {max_value}.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")
            
def read_and_process_movie_csv(movie_file_path):
    """
    Reads a movie CSV file, selects specific columns related to genres,
    and creates a 2D NumPy array with column names as the first row.

    Parameters:
    - file_path (str): The path to the movie CSV file.

    Returns:
    - genre_matrix (numpy.ndarray): 2D array containing column names and selected data.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(movie_file_path)

    # Select specific columns
    selected_columns = df[["rotten_tomatoes_link","Action & Adventure","Comedy","Drama","Science Fiction & Fantasy","Romance","Classics",
                        "Kids & Family","Mystery & Suspense","Western","Art House & International","Horror",
                        "Faith & Spirituality","Animation","Documentary","Special Interest",
                        "Musical & Performing Arts","Sports & Fitness","Television","Cult Movies",
                        "Anime & Manga","Gay & Lesbian"]]

    # Insert column names as the first row in the array
    selected_columns_array_with_header = selected_columns.columns.to_numpy()[None]
    selected_columns_2d_array = selected_columns.values

    # Combine column names and data
    genre_matrix = np.vstack([selected_columns_array_with_header, selected_columns_2d_array])

    return genre_matrix

def create_ratings_matrix(reviews_file_path):
    """
    Reads a movie review dataset from a CSV file, organizes the data into a dictionary,
    finds unique movies, and creates a 2D array representing user ratings for each movie.

    Parameters:
    - csv_file_path (str): The file path of the CSV file containing movie reviews.

    Returns:
    - list of lists: A 2D array (ratings_matrix) representing user ratings for each movie.
    """
    # Read the CSV file
    df = pd.read_csv(reviews_file_path)

    # Create a dictionary to organize data
    movie_ratings_dict = {}

    # Populate the dictionary
    for index, row in df.iterrows():
        user_name = row['Source']
        movie_id = row['Target']
        rating = row['review_score_scaled']

        if user_name not in movie_ratings_dict:
            movie_ratings_dict[user_name] = {}

        movie_ratings_dict[user_name][movie_id] = rating

    # Find all unique movies
    unique_movies = set(df['Target'])

    # Convert the dictionary to a 2D array
    num_rows = len(unique_movies)
    num_cols = len(movie_ratings_dict) + 1  # +1 for user_name column

    ratings_matrix = [[0.0] * num_cols for _ in range(num_rows)]

    row = 0
    for movie_id in unique_movies:
        ratings_matrix[row][0] = movie_id
        for col, user_name in enumerate(movie_ratings_dict, start=1):
            rating = movie_ratings_dict[user_name].get(movie_id, 0.0)
            ratings_matrix[row][col] = rating
        row += 1

    # Add column headers to the array
    column_headers = ['MovieID'] + list(movie_ratings_dict.keys())
    ratings_matrix = [column_headers] + ratings_matrix

    return ratings_matrix

def modify_binary_ratings_matrix(ratings_matrix):
    """
    Modifies the values in the ratings_matrix based on certain conditions.

    Args:
    - ratings_matrix: 2D array of numerical values

    Returns:
    - Modified binary ratings_matrix
    """
    for row in range(1, len(ratings_matrix)):  # Start from 1 to skip the header row
        for col in range(1, len(ratings_matrix[row])):
            if ratings_matrix[row][col] > 6:
                ratings_matrix[row][col] = 1
            elif ratings_matrix[row][col] != 0:
                ratings_matrix[row][col] = -1
    
    return ratings_matrix

def dot_product(genre_matrix, ratings_matrix):
    """
    Calculates the dot product between the genre_matrix and ratings_matrix 

    Args:
    - genre_matrix: 2D array of numerical values
    - ratings_matrix: 2D array of numerical values

    Returns:
    - list of lists: A 2D array (result_matrix) representing the dot product.
    """
    # Exclude the first row and the first column from both matrices
    genre_matrix_modified = [row[1:] for row in genre_matrix[1:]]
    ratings_matrix_modified = [row[1:] for row in ratings_matrix[1:]]  # Assuming the matrix has 6891 columns

    # Transpose the genre_matrix
    genre_matrix_transposed_modified = list(map(list, zip(*genre_matrix_modified)))

    # Perform the dot product
    result_matrix = [
        [
            sum(x * y for x, y in zip(row_genre, row_ratings))
            for row_ratings in ratings_matrix_modified
        ]
        for row_genre in genre_matrix_transposed_modified
    ]

    # Include labels in the result_matrix
    result_matrix = [ratings_matrix[0][1:]] + result_matrix
    result_matrix = [[genre_matrix[0][i]] + row for i, row in enumerate(result_matrix)]

    return result_matrix


def convert_to_binary(matrix):
    """
    Convert numerical values in a matrix to 1 if greater than 0, else 0, in-place.
    
    Parameters:
    - matrix: A 2D matrix (list of lists) containing numerical values.
    
    Modifies the input matrix to a binary matrix.
    """
    for row in range(1,len(matrix)):
        for col in range(1,len(matrix[row])):
            if matrix[row][col] > 0:
                matrix[row][col] = 1
            else:
                matrix[row][col] = 0
    return matrix

def calculate_euclidean_distances(target_user_id, result_matrix):
    """
    Calculate Euclidean distances between the target user and other users.

    Parameters:
    - target_user_id (int): The ID of the target user.
    - result_matrix (list): A matrix containing user IDs, X coordinates, and Y coordinates.

    Returns:
    - distances_matrix (list): A list of tuples containing distances and corresponding user IDs.
    """
    distances_matrix = []

    for i in range(1, len(result_matrix[0])):
        if i != target_user_id:
            distance = math.sqrt(
                (result_matrix[1][target_user_id] - result_matrix[1][i]) ** 2 +
                (result_matrix[2][target_user_id] - result_matrix[2][i]) ** 2
            )
            distances_matrix.append((distance, result_matrix[0][i]))

    return distances_matrix

def get_similar_users(distances_matrix, k_users):
    """
    Get a list of similar users based on the distances matrix.

    Parameters:
    - distances_matrix (list): A list of tuples containing distances and corresponding user names.
    - k_users (int): The number of similar users to retrieve.

    Returns:
    - list: A list of user names for the first k similar users.
    """
    same_users = []

    # Iterate through the range of the minimum between k_users and the length of distances_matrix
    for i in range(min(k_users, len(distances_matrix))):
        distance, user = distances_matrix[i]
        # Uncomment the line below if you want to print the distance between target_user and each similar user       
        same_users.append(user)

    return same_users



def get_recommended_movies(ratings_matrix, target_user, similar_users):
    """
    Get a list of recommended movies

    Parameters:
    - ratings_matrix: 2D array of numerical values
    - target_user: The user who will get the final recommendation
    - similar_users: A list of user names for the first k similar users.

    Returns:
    - list: A list of recommended movies for the target user
    """
    movies_recommended = []

    # Get the index of the target_user in the ratings_matrix
    user_index = ratings_matrix[0].index(target_user)

    # Get the index of the similar_users in the ratings matrix
    user_indices = [ratings_matrix[0].index(user) for user in similar_users]

    # Find movies with a rating of 1 for each user in similar_users
    for user_index in user_indices:
       #user = ratings_matrix[0][user_index]
       for movie_row in ratings_matrix[1:]:
            movie = movie_row[0]
            rating = movie_row[user_index]
            if rating == 1:
                movies_recommended.append(movie)
                
    movies_recommended = list(set(movies_recommended))

    # Remove movies that the target_user has already rated
    for row in ratings_matrix:
        if row[user_index] == 1 and row[0] in movies_recommended:
            movies_recommended.remove(row[0])

    return movies_recommended

def evaluation_metrics (movies, ratings_matrix, target_user):
    # Get the index of the target_user in the ratings_matrix
    user_index = ratings_matrix[0].index(target_user)
    
    total_items = len(movies)

    relevant_items = 0
    non_relevant_items_not_recommended = 0
    relevant_items_recommended = 0
    for row in ratings_matrix:
        if row[user_index] == 1:
            relevant_items = relevant_items +1
            if row[0] in movies:
               relevant_items_recommended =relevant_items_recommended +1
        else:
            if row[0] not in movies:
                non_relevant_items_not_recommended = non_relevant_items_not_recommended + 1
           
    
    precision = relevant_items_recommended/ float(total_items)
    recall  = relevant_items_recommended/ float(relevant_items)
    f1 = 2 *(precision * recall) / (precision + recall)
    accurancy = (float(relevant_items_recommended) + non_relevant_items_not_recommended)/17255
    
    
    
    evaluation_list = [precision,recall,f1,accurancy]
    
    return evaluation_list


def main():
    # Get the user ID
    target_user_id = get_user_input("Give the id of the user that you want to recommend a movie. Please enter a number between 1 and 17255: ", 1, 17255)
    
    # Get the total number of users for recommendation
    k_users = get_user_input("Give the total of users used for the recommendation. Please enter a number between 1 and 17254: ", 1, 17254)
    
    
    current_directory = os.getcwd()
    # Get the genre matrix
    file_path = os.path.join(current_directory,"..","data", 'movies.csv')
    genre_matrix = read_and_process_movie_csv(file_path)
    
    # Get the ratings matrix
    file_path = os.path.join(current_directory,"..","data", 'reviews.csv')
    ratings_matrix = create_ratings_matrix(file_path)
    
    #make the ratings matix binary
    ratings_matrix = modify_binary_ratings_matrix(ratings_matrix)
    
    
    #calulate the dot product
    result_matrix = dot_product(genre_matrix, ratings_matrix)
    #make the dot product matrix binary
    result_matrix = convert_to_binary(result_matrix)
    
    
    # The name of the target user
    target_user = result_matrix[0][target_user_id]
    
    
    # Save (distance, user) in a tuple for each user
    distances_matrix = calculate_euclidean_distances(target_user_id, result_matrix)
    
    # Sort distances by ascending order based on the first element (distance)
    distances_matrix = sorted(distances_matrix, key=lambda x: x[0])
    
      
    # Get similar top k users depending on the genre they watch     
    similar_users = get_similar_users(distances_matrix, k_users)      
            
    # Get the final list of the movies for recommend to the target user.
    movies_recommended =get_recommended_movies(ratings_matrix, target_user, similar_users)
    
           
    print(f"Final movies for recommendation for {target_user}:")
    print(movies_recommended)

    print("Evaluation")
    evaluation_list = evaluation_metrics(movies_recommended,ratings_matrix,target_user)
    print(evaluation_list)
    for i in evaluation_list:
        print(i)
    
    # Record the end time
    end_time = time.time()
    
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    
    # Print the result
    print(f"Elapsed time: {elapsed_time} seconds")


if __name__ == "__main__":
    main()


