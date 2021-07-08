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
#from sklearn.model_selection import train_test_split
from lenskit.batch import predict
from lenskit.metrics.predict import rmse, mae
from lenskit.crossfold import partition_users, LastFrac
from utils import read_dataset, get_grid
from gridsearch import gs, get_algo  

# Bias = basic.Bias(damping=5)
# Pop = basic.Popular()
# II = item_knn.ItemItem(20, save_nbrs=2500)
# UU = user_knn.UserUser(20)
# ALS = als.BiasedMF(50)
# IALS = als.ImplicitMF(50)
# SVD = funksvd.FunkSVD(50)
# algos = [Bias, Pop, II, UU, ALS, SVD]
names = ['Bias','Pop','II','UU','BiasedMF','SVD']
dataset = 'ML-100k'


def evaluate(aname, algo, train, test):
    fittable = util.clone(algo)
    fittable = Recommender.adapt(fittable)
    fittable.fit(train)
    users = test.user.unique()
    # now we run the recommender
    recs = batch.recommend(fittable, users, 100)
    preds = np.array((1,1))
    if aname == 'Pop':
        algo = basic.PopScore()
        fittable = util.clone(algo)
        fittable = Recommender.adapt(fittable)
        fittable.fit(train)
        preds = predict(fittable, test)
        preds['Algorithm'] = aname
    else: 
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
    plt.savefig(r"../Figures/{}_{}_gs.png".format(metric, dataset))
    plt.show()

def main(dataset):
    data, start, end = read_dataset(dataset)
    grid = get_grid(dataset)
    g = data.groupby(pd.Grouper(key='timestamp', freq='Y'))
    splits = [group for _,group in g]
    new_df = pd.DataFrame(columns = ['user', 'item', 'rating'])
    all_results = pd.DataFrame(columns = ['ndcg', 'recall', 'precision'])
    pred_results = np.array([['Algorithm','RMSE', 'MAE']])
    i = 0
    for df in splits:
        all_recs = []
        #all_preds = []
       # df = df.drop('timestamp', axis=1)
        new_df = pd.merge(new_df, df, how='outer')
        if new_df.shape[0] > 500: 
            i += 1
            #train, test = train_test_split(new_df, test_size=0.2, random_state = 42)
            tp, tp2 = partition_users(new_df, 2, method=LastFrac(0.1, col='timestamp'))
            for name in names: 
                if name == 'Pop':
                    best_para = 0
                else:
                    grid_algo = [int(s) for s in grid[name].split(',')]
                    best_para, _ = gs(name, grid_algo, tp.train)
                print(name + str(best_para))
                algo = get_algo(name, best_para)
                recs, preds = evaluate(name, algo, tp.train, tp.test)
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
            results = rla.compute(all_recs, tp.test)
            results = results.groupby('Algorithm').mean()
            all_results = all_results.append(results)

    all_results_pred = pd.DataFrame(data=pred_results[1:,1:].astype(np.float64), index = pred_results[1:,0], columns = pred_results[0,1:])
    
    all_results.to_csv(r"C:\Users\tsche\Desktop\Siegen\Results\result_{dataset}_1.csv".format(dataset=dataset))
    all_results_pred.to_csv(r"C:\Users\tsche\Desktop\Siegen\Results\result_{dataset}_2.csv".format(dataset=dataset))

    final_plot(all_results,'recall', names, start, end, i,dataset)
    final_plot(all_results,'precision', names, start, end, i,dataset)
    final_plot(all_results,'ndcg', names, start, end, i,dataset)
    final_plot(all_results_pred,'RMSE', names, start, end, i,dataset)
    final_plot(all_results_pred,'MAE', names, start, end, i,dataset)
    return all_results, all_results_pred
 
#%%
if __name__ == '__main__':
        all_results, all_results_pred = main(dataset)
