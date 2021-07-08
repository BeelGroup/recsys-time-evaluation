# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:43:26 2021

@author: tsche
"""

import pandas as pd
from lenskit import batch, topn, util
from lenskit.algorithms import item_knn, user_knn, als
from lenskit.algorithms import basic, Recommender, funksvd
from lenskit.crossfold import partition_users, SampleFrac

def evaluate(aname, algo, train, test):
    fittable = util.clone(algo)
    fittable = Recommender.adapt(fittable)
    fittable.fit(train)
    users = test.user.unique()
    # now we run the recommender
    recs = batch.recommend(fittable, users, 20)
    # add the algorithm name for analyzability
    recs['Algorithm'] = aname
    return recs


def gs(name, parameters, data):
    results = []
    if name == 'Pop':
        algo = basic.Popular()
        best_para = 0
        return best_para, 0
    for para in parameters:
        if name == 'II':
            algo = item_knn.ItemItem(para)
        elif name == 'UU':
            algo = user_knn.UserUser(para)
        elif name == 'Bias':
            algo = basic.Bias(damping=para)
        elif name == 'BiasedMF':
            algo = als.BiasedMF(para)
        elif name == 'SVD':
            algo = funksvd.FunkSVD(para)
        #print('Testing' + str(para))
        all_recs = []
        test_data = []
        version = str(para)
        for train, test in partition_users(data, 5, SampleFrac(0.2)):
            test_data.append(test)
            all_recs.append(evaluate(version, algo, train, test))
        all_recs = pd.concat(all_recs, ignore_index=True)
        test_data = pd.concat(test_data, ignore_index=True)
        rla = topn.RecListAnalysis()
        rla.add_metric(topn.ndcg)
        result = rla.compute(all_recs, test_data)
        result = result.groupby('Algorithm').ndcg.mean()
        results.append(result)
           
    results = pd.concat(results)
    idx = results.idxmax()
    best_para = int(idx)
    return best_para, results        

def get_algo(name, para):
    if name == 'II':
        algo = item_knn.ItemItem(para)
    elif name == 'UU':
        algo = user_knn.UserUser(para)
    elif name == 'Bias':
        algo = basic.Bias(damping=para)
    elif name == 'BiasedMF':
        algo = als.BiasedMF(para)
    elif name == 'SVD':
        algo = funksvd.FunkSVD(para)
    elif name == 'Pop':
        algo = basic.Popular()
    return algo

       
                
if __name__ == '__main__':
    data = pd.read_table(r"..\Datasets\Amazon\ratings_Amazon_Instant_Video.csv",
                     sep=',', header = 0, names=['user', 'item', 'rating', 'timestamp'], engine='python') 
    data['user'] = data.groupby(['user']).ngroup()
    data['item'] = data.groupby(['item']).ngroup()
    data.timestamp = pd.to_datetime(data.timestamp, unit='s', origin='1970-01-01')
    data = data.groupby("user").filter(lambda grp: len(grp) > 2)
    for algo in ['II','UU','BiasedMF','SVD']:
        best_para, results = gs(algo, [5, 10, 20, 30, 40, 50], data)
        print(algo, results)