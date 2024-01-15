## Revolutionizing Movie Recommendations for Reviewers: Unleashing the Power of Graph-Based Recommender Systems

@Gallos Konstantinos
@Polychronidis Nikolaos
@Chirtoglou Marios

Department of Informatics Aristotle University of Thessaloniki

----

This survey investigates the effectiveness of three distinct algorithms in the realm of movie recommendation systems tailored for reviewers. Addressing the challenge of providing targeted movie suggestions to reviewers for review purposes, we analyze and compare the performance of the selected algorithms. We make usage of movie and review features and project their relations between entities into a heterogeneous graph. Our findings show that there are numerous ways of approaching this problem and solving it efficiently. Moreover, they offer valuable insights for the continued improvement of recommendation strategies in the context of user-generated movie reviews. Finally, we point out some important future extensions for each of our methods.

----

In `archive rotten tomatoes` we store the raw data obtained from [Rotten Tomatoes movies and critic reviews datase](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset/data?select=rotten_tomatoes_critic_reviews.csv). There are 2 files `rotten_tomatoes_critic_reviews.csv` and `rotten_tomatoes_movies.csv`, but the first one is too big to be uploaded in this repository.

`pre-processing` directory contains the code to clean and prepare the raw data in an appropriate format. It also has the Exploratory Data Analysis on the created graph.

In `data` directory we have the preprocessed and clean data. It contains `movies.csv` (movie feature matrix) and `reviews.csv` (edge features and weights).  `reviews_clean.csv` file refers to the heterogeneous graph that does not contain any movie without features.

The algorithms we implemented are located in folders `algorithm1`, `algorithm2`, `algorithm3`. In case there is a problem running `GNN.ipynb` (`torch-sparse` package is not installed, run this code in google colab or contact us). 
