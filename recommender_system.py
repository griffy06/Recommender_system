import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# loading the data
dataset = pd.read_csv('u.data',sep='\t',names=['user_id','item_id','rating','timestamp'])
print(dataset.head(20))
print("\n\n")

# merging the movie titles with the data
movie_titles = pd.read_csv('Movie_Id_Titles')
print(movie_titles.head(20))
print("\n\n")

# since the movieId columns are same , merge the datasets with this column
dataset = pd.merge(dataset,movie_titles,on='item_id')
print(dataset.head(20))
print("\n\n")

print(dataset.describe())
print("\n\n")
# Let’s now create a dataframe with the average rating for each movie and the number of ratings.
# We are going to use these ratings to calculate the correlation between the movies later.
# Correlation is a statistical measure that indicates the extent to which two or more variables fluctuate together.
# Movies that have a high correlation coefficient are the movies that are most similar to each other.
# In our case we shall use the Pearson correlation coefficient. This number will lie between -1 and 1.
# 1 indicates a positive linear correlation while -1 indicates a negative correlation.
# 0 indicates no linear correlation. Therefore movies with a zero correlation are not similar at all.
# In order to create this dataframe we use pandas groupby functionality.
# We group the dataset by the title column and compute its mean to obtain the average rating for each movie.

ratings = pd.DataFrame(dataset.groupby('title')['rating'].mean())
print(ratings.head(20))
print("\n\n")

# We need to create a number_of_ratings column to set a threshold minimum number of ratings that a movie must have in order to be recommended
# Since it is quite possible that only one user rates 5 stars to the movie. In that case it becomes necessary to collect the statistical data
ratings['number_of_ratings'] = dataset.groupby('title')['rating'].count()
print(ratings.head(20))
print("\n\n")


ratings['rating'].hist(bins=50)
plt.show()
ratings['number_of_ratings'].hist(bins=50)
plt.show()
# see the realation between the rating of a movie and the number of ratings it has got , the graph approximately shows a linear relationship between the two
sns.jointplot(x='rating',y='number_of_ratings',data=ratings)
plt.show()

# converting the dataset into a matrix with movie titles as the columns and user ids as the rows with the data of movie ratings
movie_matrix = dataset.pivot_table(index='user_id',columns='title',values='rating')
print(movie_matrix.head(20))
print("\n\n")

# sort the movies according to the number of ratings it has in descending order
print(ratings.sort_values('number_of_ratings',ascending=False).head(10))
print("\n\n")

# Let’s assume that a user has watched Air Force One (1997) and Contact (1997).
# We would like like to recommend movies to this user based on this watching history.
# The goal is to look for movies that are similar to Contact (1997) and Air Force One (1997)
# which we shall recommend to this user. We can achieve this by computing the correlation
# between these two movies’ ratings and the ratings of the rest of the movies in the dataset.
# The first step is to create a dataframe with the ratings of these movies from our movie_matrix
AFO_user_rating = movie_matrix['Air Force One (1997)']
contact_user_rating = movie_matrix['Contact (1997)']
print(AFO_user_rating.head())
print("\n\n")
print(contact_user_rating.head())
print("\n\n")

# In order to compute the correlation between two dataframes we use pandas corrwith functionality.
# Corrwith computes the pairwise correlation of rows or columns of two dataframe objects.
# We will use this functionality to get the correlation between each movie's rating and the ratings of
# the Air Force One movie.
similar_to_AFO = movie_matrix.corrwith(AFO_user_rating)
print(similar_to_AFO.head())
print("\n\n")
# similarly for contact
similar_to_contact = movie_matrix.corrwith(contact_user_rating)
print(similar_to_contact.head())
print("\n\n")

# As noticed earlier our matrix had very many missing values since not all the movies were rated by all the users.
# We therefore drop those null values and transform correlation results into dataframes to make the results
# look more appealing.
corr_AFO = pd.DataFrame(similar_to_AFO,columns=['Correlation'])
corr_AFO.dropna(inplace=True)
print(corr_AFO.head())
print("\n\n")

corr_contact = pd.DataFrame(similar_to_contact,columns=['Correlation'])
corr_contact.dropna(inplace=True)
print(corr_contact.head())
print("\n\n")

# we have a challenge in that some of the movies have very few ratings and may end up being recommended
# simply because one or two people gave them a 5 star rating. We can fix this by setting a threshold for
# the number of ratings. From the histogram earlier we saw a sharp decline in number of ratings from 100.
# In order to do this we need to join the two dataframes with the number_of_ratings column in the ratings dataframe.
corr_AFO = corr_AFO.join(ratings['number_of_ratings'])
corr_contact = corr_contact.join(ratings['number_of_ratings'])
print(corr_AFO.head())
print("\n\n")
print(corr_contact.head())
print("\n\n")

# We shall now obtain the movies that are most similar to Air Force One (1997) by limiting them to movies that have at least 100 reviews.
# We then sort them by the correlation column and view the first 10.
print(corr_AFO[corr_AFO['number_of_ratings'] > 100].sort_values(by='Correlation', ascending=False).head(10))
print("\n\n")
# Similarly for contact
print(corr_contact[corr_contact['number_of_ratings'] > 100].sort_values(by='Correlation', ascending=False).head(10))
print("\n\n")
