# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 10:56:43 2021

@author: tsche
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import surprise
from surprise import SVD
from surprise import KNNBasic
from surprise import KNNBaseline
from surprise import KNNWithMeans
from surprise import CoClustering
from surprise import BaselineOnly
from surprise import SlopeOne
from surprise import Dataset
from surprise.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from surprise import accuracy



# load ratings form movielens dataset
#data = pd.read_table(r"C:\Users\tsche\Desktop\Siegen\Datasets\Movielens\ml-latest-small\ml-latest-small\ratings.csv",sep=',', header = 0, names=['userId', 'movieId', 'rating', 'timestamp'], engine='python') 
#data = data.sample(frac=0.05)

data = pd.read_table(r"C:\Users\tsche\Desktop\Siegen\Datasets\Movielens\ml-10M100K\ratings.dat",sep='::', header = 0, names=['userId', 'movieId', 'rating', 'timestamp'], engine='python')
#data = data.sample(frac=0.1)
#data.userId = [hash(uid) for uid in data.userId]
#data.movieId = [hash(uid) for uid in data.movieId]

# convert timestamp into date
data.timestamp = pd.to_datetime(data.timestamp, unit='s', origin='1970-01-01')

# split dataset into timeframes
g = data.groupby(pd.Grouper(key='timestamp', freq='W'))
splits = [group for _,group in g]

algos = ( BaselineOnly())#, KNNBasic(), KNNWithMeans())
reader = surprise.reader.Reader()
# apply on all data up to a timepoint
all_rmse =[]
for algo in algos:
    rmse = []
    old_df = pd.DataFrame(columns = ['userId', 'movieId', 'rating'])
    for df in splits:
        df = df.drop('timestamp', axis=1)
        new_df = pd.merge(old_df, df, how='outer')
        if new_df.shape[0] > 100:
            train, test = train_test_split(new_df, test_size=0.2, random_state = 42)
            train = Dataset.load_from_df(train, reader)
            test = Dataset.load_from_df(test, reader)
            train = train.build_full_trainset()
            test = test.build_full_trainset().build_testset()
            algo.fit(train)
            predictions = algo.test(test)
            rmse1 = accuracy.rmse(predictions, verbose=True)
           # metrics = cross_validate(algo, split, measures=['RMSE', 'MAE'], cv=2, verbose=False)
            #rmse.append(metrics["test_rmse"][-1])
            rmse.append(rmse1)
        old_df = new_df
    print(algo, 'done')
    all_rmse.append(rmse)

## plot rmse and mae over time
x = np.linspace(1995,1998,32)
plt.plot(x,all_rmse[0],x,all_rmse[1],x,all_rmse[2],x,all_rmse[3])
plt.legend(('SVD','Baseline', 'KNN', 'KNNMeans'))
plt.xlabel('year')
plt.ylabel('RMSE')
plt.title('RMSE of recommendations depending on time - Movielens')