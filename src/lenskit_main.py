# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 16:08:45 2021

@author: tsche
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lenskit import batch, topn, util
from lenskit.algorithms import item_knn, user_knn, als
from lenskit.algorithms import basic, Recommender, funksvd
from sklearn.model_selection import train_test_split
from lenskit.batch import predict
from lenskit.metrics.predict import rmse, mae

def read_dataset(name, frac=None):
    
    """ loading of different pre-downloaded datasets"""
    
    if name == 'ML-100k':
        data = pd.read_table(r"..\Datasets\Movielens\ml-100k\u.data", 
                             sep='\t', header = 0, names=['user', 'item', 'rating', 'timestamp'], engine='python')
        start = 1995
        end = 1998
        
    elif name == 'ML-1M':
        data = pd.read_table(r"..\Datasets\Movielens\ml-1m\ratings.dat", 
                             sep='::', header = 0, names=['user', 'item', 'rating', 'timestamp'], engine='python')
        start = 2000
        end = 2003
        
    elif name == 'ML-10M':
        data = pd.read_table(r"..\Datasets\Movielens\ml-10M100K\ratings.dat", 
                             sep='::', header = 0, names=['user', 'item', 'rating', 'timestamp'], engine='python')
        start = 1996
        end = 2007
        
    elif name == 'ML-20M':
        data = pd.read_table(r"..\Datasets\Movielens\ml-20m\ratings.csv", 
                             sep='::', header = 0, names=['user', 'item', 'rating', 'timestamp'], engine='python')
        start = 1996
        end = 2007 #check
        
    elif name == 'ML-100k-latest':
        data = pd.read_table(r"..\Datasets\Movielens\ml-latest-small\ratings.csv", 
                             sep=',', header = 0, names=['user', 'item', 'rating', 'timestamp'], engine='python')
        start = 1995
        end = 2017
        
    elif name == 'amazon-instantvideo':
        data = pd.read_table(r"..\Datasets\Amazon\ratings_Amazon_Instant_Video.csv",
                     sep=',', header = 0, names=['user', 'item', 'rating', 'timestamp'], engine='python') 
        data.user = [hash(uid) for uid in data.user]
        data.item = [hash(uid) for uid in data.item]
        
        start = 2007
        end = 2014
        
    elif name == 'amazon-books':
        data = pd.read_table(r"..\Datasets\Amazon\ratings_Books.csv",
                     sep=',', header = 0, names=['user', 'item', 'rating', 'timestamp'], engine='python') 
        data.user = [hash(uid) for uid in data.user]
        data.item = [hash(uid) for uid in data.item]
        
        start = 1997
        end = 2013
        
        
    else:
        raise ValueError('Dataset not implemented')
    if frac is not None:    
        data = data.sample(frac = frac)
    
    return data, start, end

Bias = basic.Bias(damping=5)
Pop = basic.Popular()
II = item_knn.ItemItem(20, save_nbrs=2500)
UU = user_knn.UserUser(30)
ALS = als.BiasedMF(50)
IALS = als.ImplicitMF(50)
SVD = funksvd.FunkSVD(50)
algos = [Bias, Pop, II, UU, ALS, SVD]
names = ['Bias', 'Pop', 'II', 'UU', 'ALS', 'SVD']

def evaluate(aname, algo, train, test):
    fittable = util.clone(algo)
    fittable = Recommender.adapt(fittable)
    fittable.fit(train)
    users = test.user.unique()
    # now we run the recommender
    recs = batch.recommend(fittable, users, 100)
    preds = np.array((1,1))
    if aname != 'Pop':
        preds = predict(fittable, test)
        preds['Algorithm'] = aname
    # add the algorithm name for analyzability
    recs['Algorithm'] = aname
    return recs, preds

def final_plot(results, metric, algos, start, end, steps, dataset):
    x = np.linspace(start, end, steps)
    for algo in algos:  
        plt.plot(x, results.loc[algo,metric])
    plt.legend(algos)
    plt.title('{} over time - {}'.format(metric, dataset))
    plt.show()
#%%
if __name__ == '__main__':
    dataset = 'ML-100k-latest'
    data, start, end = read_dataset(dataset)
    data.timestamp = pd.to_datetime(data.timestamp, unit='s', origin='1970-01-01')
    g = data.groupby(pd.Grouper(key='timestamp', freq='M'))
    splits = [group for _,group in g]
    new_df = pd.DataFrame(columns = ['user', 'item', 'rating'])
    all_results = pd.DataFrame(columns = ['ndcg', 'recall', 'precision'])
    pred_results = np.array([['Algorithm','RMSE', 'MAE']])
    i = 0
    for df in splits:
        all_recs = []
        all_preds = []
        df = df.drop('timestamp', axis=1)
        new_df = pd.merge(new_df, df, how='outer')
        if new_df.shape[0] > 100: 
            i += 1
            train, test = train_test_split(new_df, test_size=0.2, random_state = 42)
            
            for algo, name in zip(algos, names): 
                recs, preds = evaluate(name, algo, train, test)
                all_recs.append(recs)
                if preds.shape[0] > 3:
                    RMSE = rmse(preds['prediction'], preds['rating'])
                    MAE = mae(preds['prediction'], preds['rating'])
                    pred_results = np.vstack((pred_results, [name, RMSE.astype(np.float64), MAE.astype(np.float64)]))
            
            all_recs = pd.concat(all_recs, ignore_index=True)
    
            rla = topn.RecListAnalysis()
            rla.add_metric(topn.ndcg)
            rla.add_metric(topn.recall)
            rla.add_metric(topn.precision)
            results = rla.compute(all_recs, test)
            results = results.groupby('Algorithm').mean()
            all_results = all_results.append(results)

    all_results_pred = pd.DataFrame(data=pred_results[1:,1:].astype(np.float64), index = pred_results[1:,0], columns = pred_results[0,1:])
    
    final_plot(all_results,'recall', names, start, end, i,dataset)
    final_plot(all_results,'precision', names, start, end, i,dataset)
    final_plot(all_results,'ndcg', names, start, end, i,dataset)
    names = ['Bias','II', 'UU', 'ALS', 'SVD']
    final_plot(all_results_pred,'RMSE', names, start, end, i,dataset)
    final_plot(all_results_pred,'MAE', names, start, end, i,dataset)
