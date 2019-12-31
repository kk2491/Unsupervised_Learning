import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":

	data = pd.read_csv("plums.csv")

	# Remove the serial number
	data = data.values[:,1:]
	print(data.shape)

	X = np.log(1.0/data)
	print(X.T.shape)
	w1 = np.arange(1100, 2300, 2)

	fig = plt.figure(figsize = (8, 6))

	with plt.style.context(("ggplot")):

		plt.plot(w1, X.T)
		plt.xlabel("Wavelength (nm)")
		plt.ylabel("Absorbance spectra")
		plt.show()


	pca = PCA()

	T = pca.fit_transform(StandardScaler().fit_transform(X))

	fig = plt.figure(figsize = (8, 6))

	with plt.style.context(("ggplot")):
		plt.scatter(T[:, 0], T[:, 1], edgecolors = "k", cmap = "jet")
		plt.xlabel("PC1")
		plt.ylabel("PC2")
		plt.title("Score plot")
		plt.show()


	fig = plt.figure(figsize = (8, 6))
	with plt.style.context(("ggplot")):
		plt.scatter(T[:, 0], T[:, 1], edgecolors = "k", cmap = "jet")
		plt.xlim(-60, 60)
		plt.ylim(-60, 60)
		plt.xlabel("PC1")
		plt.ylabel("PC2")
		plt.title("Score plot")
		plt.show()
		
