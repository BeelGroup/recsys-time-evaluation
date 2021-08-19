# Time dependend evalaution of Recommender Systems

This repository holds all relevant code to evalaute recsys algorithms over time. The main code is in the file
lenskit_gs_main.py. The algorithms and evaluation are from the Lenskit libary. 

Dependencies:
pandas
numpy
sklearn
tensorflow
lenskit

The following datasets were used:
Neflix
Amazon (books, instant video, toys and games, Music)
Movielens (100k, 1M, 10M)
yelp
Any other datasets that include a timestamp can be used. The reading instructions just need to be added
to the function 'read_datasets' in the util.py file.