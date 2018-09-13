# Movie recommendation system
# Item based collaborative filtering

import pandas as pd
import numpy as np

# Sorting the data
r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('ratings.data', sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")
m_cols = ['movie_id', 'title']
movies = pd.read_csv('movies.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")
ratings = pd.merge(movies, ratings)
#print(ratings.head())

# Creating a new table to look at relationship between movies from each user_id
# We want to see what movies f.i. the user with the id 1 has rated
movieRatings = ratings.pivot_table(index=['user_id'], columns=['title'], values='rating')
#print(movieRatings.head()) #we see that the user 1 has f.i. rated '101 dalmatians' and '12 angry men' and more

# Extracting the users who rated Star Wars
starWarsRatings = movieRatings['Star Wars (1977)']
#print(starWarsRatings.head())

# Finding out the correlation of Star Wars' ratings to other movies, using corrwith
similarMovies = movieRatings.corrwith(starWarsRatings)
similarMovies = similarMovies.dropna() #Dropping missing results
df = pd.DataFrame(similarMovies)
#print(df.head())

# Sorting the results
sorted = similarMovies.sort_values(ascending=False)
#print(sorted) 

# many movies with high correlation don't seem to make sense
# the recommendation of movies for everyone who rated Star Wars high don't make sense
# Something seems to be off

# Throwing out the movies that have beeen rated just few times
movieStats = ratings.groupby('title').agg({'rating':[np.size, np.mean]}) #creating new dataframe aggregating together the number of ratings and mean of rating for every movie title
#print(movieStats.head()) #f.i. '101 Dalamatians' have 109 ratings, but a movie called '1-900' has just 5 ratings

# Throwing out movies that have been rated less than 100 times
popularMovies = movieStats['rating']['size'] >= 100
print(movieStats[popularMovies].sort_values([('rating', 'mean')], ascending=False)[:15])

# Joining together with the original data set
df = movieStats[popularMovies].join(pd.DataFrame(similarMovies, columns=['similarity']))
print(df.head()) #makes more sense

# Sorting by similarity to Star  Wars
print(df.sort_values(['similarity'], ascending=False)[:15])

# Now this looks reasonable




