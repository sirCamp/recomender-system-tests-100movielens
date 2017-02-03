from __future__ import division
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt

## Memory-Based Collaborative Filtering ##


# header of user file (user rating)
header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('ml-100k/u.data', sep='\t', names=header)

# header of item file (films)
headerm = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure','Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy','Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
dfm = pd.read_csv('ml-100k/u.item', sep='|', names=headerm)

# personal ratings
headerc = ['user_id','item_id','rating','title']
dfc = pd.read_csv('ml-100k/my_ratings.data',sep="::",names=headerc,engine='python')

# get all items and all users
n_users = df.user_id.unique().shape[0] +1
n_items = df.item_id.unique().shape[0] +1
print 'Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items)

# split data in train and test
train_data, test_data = train_test_split(df, test_size=0.10)

# Create two user-item matrices, one for training and another for testing and another for personal ratings
train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]

test_matrix_user_id = []
test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    # print str(line[0])+" "+str(line [1])+" "+str(line[2])+" "+str(line[3])
    test_data_matrix[line[1]-1, line[2]-1] = line[3]
    test_matrix_user_id.append(line[0])


# similarity between items and the other one is between users
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')


def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        # You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred



def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

def normalize_predicted_rating(predictions):

    result_renormalized = []
    for i in predictions:
        result_renormalized.append(5*i)
    return result_renormalized


def get_predicted_film_by_user(user_id,ratings):

    result = normalize_predicted_rating(ratings[user_id])#[:15]
    films = dfm.values#[:15]
    list_v = []
    idx = 0

    for f in films:
        list_v.append((result[idx],f[1]))
        idx += 1

    fresult= sorted(list_v,key=lambda x: float(x[0]),reverse=True)
    return fresult[:15]




user_prediction = predict(test_data_matrix,user_similarity,type='user')
item_prediction = predict(test_data_matrix, item_similarity, type='item')
result = user_prediction
result_it = item_prediction
result_user_id = test_matrix_user_id



user_id = 1
res = get_predicted_film_by_user(user_id,result)

print "\n\n"
print "Recommended 15 films for user "+str(user_id)+": "
idx = 1
for racc_film in res:

    print str(idx)+") "+racc_film[1]
    idx += 1


print "\n\n"

print 'User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix))
print 'Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix))


sparsity=round(1.0-len(df)/float(n_users*n_items),3)
print 'The sparsity level of MovieLens100K is ' +  str(sparsity*100) + '%'