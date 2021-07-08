# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 17:17:14 2021

@author: tsche
"""
import pandas as pd

def get_grid(name):
    grids = pd.read_excel('Grids.xls')
    grids = grids.set_index('Algo')
    
    grid = grids[name]

    return grid

def read_dataset(name, frac=None):
    
    """ loading of different pre-downloaded datasets"""
    
    if name == 'ML-100k':
        data = pd.read_table(r"..\Datasets\Movielens\ml-100k\u.data", 
                             sep='\t', header = 0, names=['user', 'item', 'rating', 'timestamp'], engine='python')
        data.timestamp = pd.to_datetime(data.timestamp, unit='s', origin='1970-01-01')
        start = 1995
        end = 1998
        
    elif name == 'ML-1M':
        data = pd.read_table(r"..\Datasets\Movielens\ml-1m\ratings.dat", 
                             sep='::', header = 0, names=['user', 'item', 'rating', 'timestamp'], engine='python')
        data.timestamp = pd.to_datetime(data.timestamp, unit='s', origin='1970-01-01')
        start = 2000
        end = 2003
                
    elif name == 'ML-10M':
        data = pd.read_table(r"..\Datasets\Movielens\ml-10M100K\ratings.dat", 
                             sep='::', header = 0, names=['user', 'item', 'rating', 'timestamp'], engine='python')
        data.timestamp = pd.to_datetime(data.timestamp, unit='s', origin='1970-01-01')
        start = 1996
        end = 2009
        
        
    elif name == 'ML-100k-latest':
        data = pd.read_table(r"..\Datasets\Movielens\ml-latest-small\ratings.csv", 
                             sep='::', header = 0, names=['user', 'item', 'rating', 'timestamp'], engine='python')
        data.timestamp = pd.to_datetime(data.timestamp, unit='s', origin='1970-01-01')
        start = 1995
        end = 2017
               
    elif name == 'amazon-instantvideo':
        data = pd.read_table(r"..\Datasets\Amazon\ratings_Amazon_Instant_Video.csv",
                     sep=',', header = 0, names=['user', 'item', 'rating', 'timestamp'], engine='python') 
        data['user'] = data.groupby(['user']).ngroup()
        data['item'] = data.groupby(['item']).ngroup()
        data.timestamp = pd.to_datetime(data.timestamp, unit='s', origin='1970-01-01')
        start = 2007
        end = 2014
        
    elif name == 'amazon-books':
        data = pd.read_table(r"..\Datasets\Amazon\ratings_Books.csv",
                     sep=',', header = 0, names=['user', 'item', 'rating', 'timestamp'], engine='python') 
        data['user'] = data.groupby(['user']).ngroup()
        data['item'] = data.groupby(['item']).ngroup()
        data.timestamp = pd.to_datetime(data.timestamp, unit='s', origin='1970-01-01')
        start = 1997
        end = 2013
        
    elif name == 'amazon-toys':
        data = pd.read_table(r"..\Datasets\Amazon\ratings_Toys_and_Games.csv",
                     sep=',', header = 0, names=['user', 'item', 'rating', 'timestamp'], engine='python') 
        data['user'] = data.groupby(['user']).ngroup()
        data['item'] = data.groupby(['item']).ngroup()
        data.timestamp = pd.to_datetime(data.timestamp, unit='s', origin='1970-01-01')
        start = 2001
        end = 2014
    
    elif name == 'amazon-electronics':
        data = pd.read_table(r"..\Datasets\Amazon\ratings_Electronics.csv",
                     sep=',', header = 0, names=['user', 'item', 'rating', 'timestamp'], engine='python') 
        data['user'] = data.groupby(['user']).ngroup()
        data['item'] = data.groupby(['item']).ngroup()
        data.timestamp = pd.to_datetime(data.timestamp, unit='s', origin='1970-01-01')
        start = 2000
        end = 2014
            
    elif name == 'amazon-music':
        data = pd.read_table(r"..\Datasets\Amazon\ratings_Digital_Music.csv",
                     sep=',', header = 0, names=['item', 'user', 'rating', 'timestamp'], engine='python') 
        data = data[['user', 'item', 'rating', 'timestamp']]
        data['user'] = data.groupby(['user']).ngroup()
        data['item'] = data.groupby(['item']).ngroup()
        data.timestamp = pd.to_datetime(data.timestamp, unit='s', origin='1970-01-01')
        start = 1998
        end = 2014
        
    elif name == 'netflix':
        data = pd.read_table(r"..\Datasets\netflix\NetflixRatings.csv", sep=",", names = ['item','user', 'rating', 'timestamp'])
        data = data[['user', 'item', 'rating', 'timestamp']]
        data.timestamp = pd.to_datetime(data.timestamp)
        start = 1998
        end = 2005
        
    elif name == 'yelp':
        data = pd.read_json(r"..\Datasets\yelp_training_set\yelp_training_set_review.json", lines=True)
        data = data.rename(columns={"user_id": "user", "business_id": "item", "stars": "rating", "date": "timestamp"})
        data = data[['user', 'item', 'rating', 'timestamp']]
        data.timestamp = pd.to_datetime(data.timestamp)
        data['user'] = data.groupby(['user']).ngroup()
        data['item'] = data.groupby(['item']).ngroup()
        start = 2006
        end = 2013
                
    elif name == 'epinions':
        data = pd.read_table(r"..\Datasets\epinions\rating_with_timestamp.txt", 
                             delim_whitespace=True, names = ['user','item','category','rating','helpfulness', 'timestamp'])
        data = data[['user', 'item', 'rating', 'timestamp']]
        data.timestamp = pd.to_datetime(data.timestamp, unit='s', origin='1970-01-01')
        start = 1999
        end = 2011
               
    else:
        raise ValueError('Dataset not implemented')
    
    data = data.groupby("user").filter(lambda grp: len(grp) > 2)

    if frac is not None:    
        data = data.sample(frac = frac)
    
    return data, start, end

