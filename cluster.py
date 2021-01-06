# Reference : https://medium.com/@franky07724_57962/using-keras-pre-trained-models-for-feature-extraction-in-image-clustering-a142c6cdf5b1
# IMP - https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/

# For mini Cat and Dog - Each class is having 25 samples
# No PCA - Features directly from VGGnet

from keras.preprocessing import image 
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess
import numpy as np 
import glob
import os
import shutil
from sklearn.cluster import KMeans 
import sys
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
from keras.applications.resnet50 import ResNet50 
from keras.applications.resnet50 import preprocess_input as resnet_preprocess
import simplejson
from collections import OrderedDict
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_preprocess
import pandas as pd 
import seaborn as sns 
import matplotlib.patches as mpatches


def read_confif_file():

	config_file = "/home/kishor/Meta_Cognition_Experiments/1_Clustering/approach_2/general_implementation/config.json"

	with open(config_file, "r") as cfile:
		config_paras = simplejson.loads(cfile.read(), object_pairs_hook = OrderedDict)

	return config_paras


# Add normalization - /255
# Normalization and PCA - only if needed
def generate_resnet_features(image_list, model, config_paras):

	source_dir = config_paras["source_dir"]
	feature_vector_path = config_paras["feature_path"] 

	resnet_feature_list = []

	print("Feature Generation Started")

	for idx, each_image in enumerate(image_list):
		img = image.load_img(each_image, target_size = (224, 224))
		img_data = image.img_to_array(img)
		img_data = np.expand_dims(img_data, axis = 0)
		img_data = resnet_preprocess(img_data)

		resnet_features = model.predict(img_data)

		resnet_features_np = np.array(resnet_features)

		resnet_feature_list.append(resnet_features_np.flatten())

	resnet_feature_list_np = np.array(resnet_feature_list)

	os.chdir(source_dir)
	np.save(feature_vector_path, resnet_feature_list_np)

	print("Feature Generation Completed and saved the features")
	return resnet_feature_list_np


# Add normalization - /255
# Normalization and PCA - only if needed
def generate_feature_vector(image_list, model, config_paras):
	
	source_dir = config_paras["source_dir"]
	feature_vector_path = config_paras["feature_path"] 

	vgg16_features_list = []

	print("Feature Generation Started")
	
	for idx, each_image in enumerate(image_list):
		#print(each_image)
		img = image.load_img(each_image, target_size = (224, 224))
		#print("Shape 1 : {}".format(img.shape))
		img_data = image.img_to_array(img)
		# newly added normalization 
		img_data = img_data/255.0
		#print("Shape 2 : {}".format(img_data.shape))
		img_data = np.expand_dims(img_data, axis = 0)
		#print("Shape 3 : {}".format(img_data.shape))
		
		#img_data = preprocess_input(img_data)
		img_data = vgg16_preprocess(img_data)

		vgg16_features = model.predict(img_data)
		vgg16_features_np = np.array(vgg16_features)

		vgg16_features_list.append(vgg16_features_np.flatten())

	
	vgg16_features_list_np = np.array(vgg16_features_list)

	os.chdir(source_dir)
	np.save(feature_vector_path, vgg16_features_list_np)

	print("Feature Generation Completed and saved the features")

	return vgg16_features_list_np


# Add normalization - /255 
# Normalization and PCA - only if needed
def generate_inception_features(image_list, model):

	inception_features_list = []
	print("Feature Generation Started")

	for idx, each_image in enumerate(image_list):

		img = image.load_img(each_image, target_size = (299, 299))
		img_data = image.img_to_array(img)
		img_data = img_data/255.0
		img_data = np.expand_dims(img_data, axis = 0)
		img_data = inception_preprocess(img_data)
		
		inception_features = model.predict(img_data)
		inception_features_np = np.array(inception_features)

		inception_features_list.append(inception_features_np.flatten())

	inception_features_list_np = np.array(inception_features_list)

	os.chdir(source_dir)
	np.save(feature_vector_path, inception_features_list_np)

	print("Feature Generation Completed and saved the features")

	return inception_features_list_np


# TODO 
def generate_pca_features(cnn_features_list_np):
	
	training_data = cnn_features_list_np
	scaler = MinMaxScaler()
	training_data_rescaled = scaler.fit_transform(training_data)
	print("Training Data Rescaled Shape : {}".format(training_data_rescaled.shape))

	pca_comp = PCA(n_components = 0.95)
	training_data_pca = pca_comp.fit_transform(training_data_rescaled)

	print("Reduced Features Shape : {}".format(training_data_rescaled.shape))

	reduced_feature_list_np = training_data_pca

	return reduced_feature_list_np


def plot_sample_scores(x, data):
	plt.bar(x-0.25, data[0], color = "b", width = 0.25)
	plt.bar(x+0.00, data[1], color = "g", width = 0.25)
	plt.bar(x+0.25, data[2], color = "r", width = 0.25)
	plt.title("Silhouette Analysis ")
	zero = mpatches.Patch(color='blue', label='zero')
	positive = mpatches.Patch(color='green', label='positive')
	negative = mpatches.Patch(color='red', label='negative')
	plt.legend(handles=[zero, positive, negative])
	plt.show()	



if __name__ == "__main__":

	# Read from JSON file
	config_paras = read_confif_file()
	print("Config parameters : {}".format(config_paras))

	img_dir 			= config_paras["dataset_dir"]
	source_dir 			= config_paras["source_dir"]
	feature_vector_path = config_paras["feature_path"]
	model_path 			= config_paras["model_path"]
	base_model 			= config_paras["base_model"]
	vector_gen 			= config_paras["vector_gen"]
	test_phase	 		= config_paras["analysis_phase"]

	curr_dir = os.getcwd()
	os.chdir(img_dir)
	image_list = sorted(glob.glob("*.jpg"))
	print("===================================")
	print("Number of images : {}".format(len(image_list)))
	print("===================================")
	os.chdir(img_dir)

	if base_model == "resnet50":
		model = ResNet50(weights = "imagenet", include_top = False)
		model.summary()

		if (vector_gen == True):
			features_list_np = generate_resnet_features(image_list, model, config_paras)
		else:
			features_list_np = np.load(feature_vector_path)

		print("Features : {}".format(len(features_list_np[0])))

	elif base_model == "vgg16":
		model = VGG16(weights = "imagenet", include_top = False)
		model.summary()

		if (vector_gen == True):
			features_list_np = generate_feature_vector(image_list, model, config_paras)
		else:
			features_list_np = np.load(feature_vector_path)

		print("VGG Features : {}".format(len(features_list_np[0])))
	
	elif base_model == "inception":
		model = InceptionV3(weights = "imagenet", include_top = False)
		model.summary()

		if vector_gen == True:
			features_list_np = generate_feature_vector(image_list, model, config_paras)
		else:
			features_list_np = np.load(feature_vector_path)

	else:
		print("No options")


	if test_phase == True:
		error = []
		score_list = []
		sample_score_list = []

		zero_list 	= []
		pos_list 	= []
		neg_list 	= []

		lower_range = config_paras["clust_lower_range"]
		upper_range = config_paras["clust_upper_range"]

		for i in range(lower_range, upper_range):
			# For Elbow curve
			kmeans_cluster = KMeans(n_clusters = i)
			kmeans_cluster_fit = kmeans_cluster.fit(features_list_np)
			loss = kmeans_cluster_fit.inertia_
			error.append(kmeans_cluster_fit.inertia_)

			# For Silhoutte analysis
			preds = kmeans_cluster.fit_predict(features_list_np)
			score = silhouette_score(features_list_np, preds)
			score_list.append(score)

			sample_score = silhouette_samples(features_list_np, preds)
			sample_score_list.append(sample_score)

			print("Cluster : {} | Loss : {} | Silhoutte Score : {}".format(i, loss, score))

			zero_samples = 0
			positive_samples = 0
			negative_samples = 0

			for each_sample in sample_score:
				if each_sample == 0:
					zero_samples += 1
				if each_sample > 0:
					positive_samples += 1
				if each_sample < 0:
					negative_samples += 1

			print("Cluster : {} | Silhouette sample distribution - Zero : {} | Positive : {} | Negative : {}".format(i, zero_samples, positive_samples, negative_samples))

			zero_list.append(zero_samples)
			pos_list.append(positive_samples)
			neg_list.append(negative_samples)

		fig, axes = plt.subplots(nrows = 1, ncols = 2)
		x_axis = range(lower_range, upper_range)
		axes[0].plot(x_axis, error)
		axes[1].plot(x_axis, score_list)
		plt.show()

		# silhouette sample score needs to be added 
		plot_data = [zero_list, pos_list, neg_list]
		x_axis_data = np.arange(lower_range, upper_range)
		plot_sample_scores(x_axis_data, plot_data)

	else:

		kmeans_cluster = KMeans(n_clusters = 14).fit(features_list_np)
		pickle.dump(kmeans_cluster, open(model_path, "wb"))

		print(kmeans_cluster)

		print("Centroid : {}".format(kmeans_cluster.cluster_centers_))
		print("Labels   : {}".format(kmeans_cluster.labels_))	
	
