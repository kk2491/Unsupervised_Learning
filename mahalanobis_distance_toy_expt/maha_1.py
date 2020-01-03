# Not working

import scipy as sc 
import numpy as np 
import pandas as pd 
from scipy.spatial.distance import mahalanobis 

datadict = {
'country': ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Ecuador', 'Colombia', 'Paraguay', 'Peru', 'Venezuela'],
'd1': [0.34, -0.19, 0.37, 1.17, -0.31, -0.3, -0.48, -0.15, -0.61],
'd2': [-0.57, -0.69, -0.28, 0.68, -2.19, -0.83, -0.53, -1, -1.39],
'd3': [-0.02, -0.55, 0.07, 1.2, -0.14, -0.85, -0.9, -0.47, -1.02],
'd4': [-0.69, -0.18, 0.05, 1.43, -0.02, -0.7, -0.72, 0.23, -1.08],
'd5': [-0.83, -0.69, -0.39, 1.31, -0.7, -0.75, -1.04, -0.52, -1.22],
'd6': [-0.45, -0.77, 0.05, 1.37, -0.1, -0.67, -1.4, -0.35, -0.89]
}

pairsdict = {
'country1': ['Argentina', 'Chile', 'Ecuador', 'Peru'],
'country2': ['Bolivia', 'Venezuela', 'Colombia', 'Peru']
}

print("===================")
print("Data  : {}".format(datadict))
print("Pairs : {}".format(pairsdict))

df = pd.DataFrame(datadict)

print("===================")
print("DataFrame : \n {}".format(df))
pairs = pd.DataFrame(pairsdict)

print("===================")
print("Pairs DataFrame : \n {}".format(pairs))

pairs = pairs.merge(df, how='left', left_on=['country1'], right_on=['country'])
print("===================")
print(pairs)

#pairs = pairs.merge(df, how='left', left_on=['country2'], right_on=['country'])
#print(pairs)

pairs['vector1'] = pairs[['d1_x','d2_x','d3_x','d4_x','d5_x','d6_x']].values.tolist()
pairs['vector2'] = pairs[['d1_y','d2_y','d3_y','d4_y','d5_y','d6_y']].values.tolist()

print("===================")
#print("pairs")

mahala = pairs[['country1', 'country2', 'vector1', 'vector2']]
 
#Calculate covariance matrix
covmx = df.cov()
invcovmx = sp.linalg.inv(covmx)
 
#Calculate Mahalanobis distance
mahala['mahala_dist'] = mahala.apply(lambda x: (mahalanobis(x['vector1'], x['vector2'], invcovmx)), axis=1)
 
mahala = mahala[['country1', 'country2', 'mahala_dist']]