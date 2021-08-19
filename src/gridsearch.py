# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:43:26 2021

@author: tsche
"""

import pandas as pd
import numpy as np
from lenskit import batch, topn, util
from lenskit.algorithms import item_knn, user_knn, als, tf
from lenskit.algorithms import basic, Recommender, funksvd
from lenskit.crossfold import partition_users, SampleFrac
from lenskit.metrics.predict import rmse

def evaluate(aname, algo, train, test):
    fittable = util.clone(algo)
    fittable = Recommender.adapt(fittable)
    fittable.fit(train)
    users = test.user.unique()
    recs = batch.recommend(fittable, users, 10)
    preds = batch.predict(fittable, test)
    preds['Algorithm'] = aname
    recs['Algorithm'] = aname
    return recs, preds


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
        elif name == 'BPR':
            algo = tf.BPR(para)
        #print('Testing' + str(para))
        all_recs = []
        test_data = []
        version = str(para)
        for train, test in partition_users(data, 3, SampleFrac(0.2)):
            test_data.append(test)
            recs, preds = evaluate(version, algo, train, test)
            all_recs.append(recs)
        all_recs = pd.concat(all_recs, ignore_index=True)
        test_data = pd.concat(test_data, ignore_index=True)
        rla = topn.RecListAnalysis()
        rla.add_metric(topn.ndcg, k=10)
        result = rla.compute(all_recs, test_data)
        result = result.groupby('Algorithm').ndcg.mean()
        results.append(result)
    results = pd.concat(results)
    #print(results)
    idx = results.idxmax()
    best_para = int(idx)
    return best_para, results        

def gs_rmse(name, parameters, data):
    results = [10]
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
        elif name == 'BPR':
            algo = tf.BPR(para)
        #print('Testing' + str(para))
        all_preds = []
        version = str(para)
        for train, test in partition_users(data, 5, SampleFrac(0.2)):
            recs, preds = evaluate(version, algo, train, test)
            all_preds.append(preds)
        all_preds = pd.concat(all_preds, ignore_index=True)
        result = rmse(all_preds['prediction'], all_preds['rating'])
        results = np.vstack((results, result))
    #print(results)
    idx = np.argmin(results)
    best_para = parameters[idx-1]
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
    elif name == 'BPR':
        algo = tf.BPR(para)
    return algo
