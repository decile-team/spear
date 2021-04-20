import apricot
import numpy as np
import torch
from sklearn.metrics.pairwise import euclidean_distances
from utils_jl import find_indices, get_similarity_kernel
#need to import utils if utils_jl is imported?

def rand_subset(n_all, n_instances):
	'''
		A function to choose random indices of the input instances to be labeled

	Args:
		n_all: number of available instances, type in integer
		n_intances: number of instances to be labelled, type is integer
	
	Return:
		A numpy.ndarray of the indices(of shape (n_sup,) and each element in the range [0,n_all-1)) to be labeled
	'''
	assert type(n_all) == np.int or type(n_all) == np.float
	assert type(n_instances) == np.int or type(n_instances) == np.float
	assert np.int(n_all) > np.int(n_instances)
	return np.random.choice(int(n_all), int(n_instances), replace = False)

def unsup_subet(x_train, n_unsup):
	'''
		A function for unsupervised subset selection
	Args:
		x_train: A numpy.ndarray of shape(n_instances, n_features).All the data, intended to be used for training
		n_unsup: number of instances to be found during unsupervised subset selection
	
	Return:
		numpy.ndarray of indices(shape is (n_sup,), each element lies in [0,x_train.shape[0])), the result of subset selection
	'''
	fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state = 0, n_samples = n_unsup)
	x_sub = fl.fit_transform(x_train)
	indices = find_indices(x_train, x_sub)
	return indices

def sup_subset(path_train, n_sup, n_classes, qc = 0.85):
	'''
		A function for supervised subset selection
	
	Args:
		path_train: path to the pickle file containing all the training data
		n_sup: number of instances to be found during supervised subset selection
		n_classes: number of classes of the training data
		qc: Quality index of shape (n_lfs,) of type numpy.ndarray OR a float. Values must be between 0 and 1. Default is 0.85

	Return:
		numpy.ndarray of indices(shape is (n_sup,), each element lies in [0,num_instances)), the result of subset selection
	'''
	assert (type(qc) == np.float and (qc >= 0 and qc <= 1)) or (type(qc) == np.ndarray and (np.all(np.logical_and(qc>=0, qc<=1)) ) )\
		 or (type(qc) == np.int and (qc == 0 or qc == 1))

	data = get_data(path_train)
	m = data[2]
	s = data[6]
	k = data[8]
	continuous_mask = data[7]
	qc_temp = torch.tensor(qc).double() if type(qc) == np.ndarray else qc
	params = torch.ones((n_classes, n_lfs)).double() # initialisation of gm parameters, refer section 3.4 in the JL paper

	y_train_pred = predict_gm(params, params, m, s, k, n_classes, continuous_mask, qc_temp)
	kernel = get_similarity_kernel(y_train_pred)
	similarity = euclidean_distances(x_train)
	sim_mat = kernel * similarity
	fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state = 0, metric = 'precomputed', n_samples = n_sup)
	sim_sub = fl.fit_transform(sim_mat)
	indices = find_indices(sim_mat, sim_sub)

	return indices