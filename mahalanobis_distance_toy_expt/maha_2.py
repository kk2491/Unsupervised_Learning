# Reference 
# https://www.machinelearningplus.com/statistics/mahalanobis-distance/



import pandas as pd 
import scipy as sp 
from scipy import linalg
import numpy as np 
import sys 
from scipy.spatial import distance


# D^2 = (x - m).T . C-1 . (x - m)
def mahalanobis(x, data, cov=None):

	x_minus_mu = x - np.mean(data)

	if not cov:
		cov = np.cov(data.values.T)

	inv_covmat = linalg.inv(cov) 

	left_term = np.dot(x_minus_mu, inv_covmat)
	mahal = np.dot(left_term, x_minus_mu.T)

	print("===================")
	print(mahal)
	print(mahal.shape)
	print("===================")

	return mahal.diagonal

if __name__ == "__main__":

	experiment = sys.argv[1]

	if experiment == "1":
		
		file_path = "diamonds.csv"
		df = pd.read_csv(file_path).iloc[:, [0,4,6]]
		print(df.shape)
		print(df.head())

		df_x = df[["carat", "depth", "price"]].head(1)
		print(df_x.shape)

		df_x['mahala'] = mahalanobis(x=df_x, data=df[['carat', 'depth', 'price']])
		df_x.head()

	if experiment == "2":
		file_path = "diamonds.csv"
		data = pd.read_csv(file_path).iloc[:, [0,4,6]]
		print("Shape of the data : {}".format(data.shape))

		cov_mat = np.cov(data.values.T)
		inv_covmat = np.linalg.inv(cov_mat)

		print("Covariance matrix shape : {}".format(cov_mat.shape))
		print("Inverse Covariance matrix shape : {}".format(inv_covmat.shape))

		test_sample = df_x = df[["carat", "depth", "price"]].head(1)

		