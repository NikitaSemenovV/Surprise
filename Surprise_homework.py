#!/usr/bin/env python
# coding: utf-8

# # Surprise homework

# In[1]:


import io 
from collections import defaultdict
from surprise import SVD
from surprise import Dataset
from surprise import KNNWithMeans
from surprise import NormalPredictor
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold


# Load the movielens-100k dataset (download it if needed),
data = Dataset.load_builtin('ml-100k')


# ### Random rating based on the distribution

# In[2]:


algo = NormalPredictor()

cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=True)


# ### kNN cosine

# In[2]:


algo = KNNWithMeans(k = 30, sim_options={'name': 'cosine'})

cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=True)


# ### kNN Mean Squared Difference

# In[3]:


algo = KNNWithMeans(k = 30, sim_options={'name': 'msd'})

cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=True)


# ### kNN Pearson

# In[7]:


algo = KNNWithMeans(k = 30, sim_options={'name': 'pearson'})

cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=True)


# ### SVD

# In[4]:


algo = SVD()

cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=True)


# ### Calculate precision@k and recall@k

# In[5]:


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls

algo = SVD()
kf = KFold(n_splits=5)

for trainset, testset in kf.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)
    precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=3.52)

    # Precision and recall can then be averaged over all users
    print(sum(prec for prec in precisions.values()) / len(precisions))
    print(sum(rec for rec in recalls.values()) / len(recalls))


# ### Predict

# In[2]:


n = 5
trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)

testset = trainset.build_anti_testset()
predictions = algo.test(testset)

top_n = defaultdict(list)
for uid, iid, true_r, est, _ in predictions:
    top_n[uid].append((iid, est))

for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]


# In[21]:


import os
import pandas as pd
file_path = os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/u.item')
films = pd.read_csv(file_path, sep="|", encoding='ansi', usecols=[0,1,2], names=['id','name', 'date'])
print("User 28")
for id, score in top_n['28']:
    name = films.at[int(id) - 1, 'name']
    date = films.at[int(id) - 1, 'date']
    print("{:<6} ({:<50} {:<11}) {:<10.3f}".format(id, name, date, score))

