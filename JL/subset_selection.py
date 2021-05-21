import apricot
import numpy as np
import torch
from sklearn.metrics.pairwise import euclidean_distances
from utils_jl import find_indices, get_similarity_kernel
import pickle

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

def unsup_subset(x_train, n_unsup):
	'''
		A function for unsupervised subset selection
	Args:
		x_train: A torch.Tensor of shape [n_instances, n_features].All the data, intended to be used for training
		n_unsup: number of instances to be found during unsupervised subset selection, type is integer
	
	Return:
		numpy.ndarray of indices(shape is (n_sup,), each element lies in [0,x_train.shape[0])), the result of subset selection
	'''
	assert x_train.shape[0] > int(n_unsup)
	assert type(x_train) == torch.Tensor
	fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state = 0, n_samples = int(n_unsup))
	x_sub = fl.fit_transform(x_train)
	indices = find_indices(x_train, x_sub)
	return indices

def sup_subset(path_train, n_sup, n_classes, qc = 0.85):
	'''
		A function for supervised subset selection whcih just returns indices
	
	Args:
		path_train: path to the pickle file containing all the training data in standard format
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

	assert m.shape[0] > int(n_sup)

	y_train_pred = predict_gm(params, params, m, s, k, n_classes, continuous_mask, qc_temp)
	kernel = get_similarity_kernel(y_train_pred)
	similarity = euclidean_distances(x_train)
	sim_mat = kernel * similarity
	fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state = 0, metric = 'precomputed', n_samples = n_sup)
	sim_sub = fl.fit_transform(sim_mat)
	indices = find_indices(sim_mat, sim_sub)

	return indices

def sup_subset_adv(path_train, path_save_L, path_save_U, n_sup, n_classes, qc = 0.85):
	'''
		A function for supervised subset selection which makes separate pickle files of data, one for those to be labelled, other that can be left unlabelled
	
	Args:
		path_train: path to the pickle file containing all the training data in standard format
		path_save_L: path to save the pickle file of set of instances to be labelled. Note that instances are not labelled yet. Extension should be .pkl
		path_save_U: path to save the pickle file of set of instances that can be left unlabelled. Extension should be .pkl
		n_sup: number of instances to be found during supervised subset selection
		n_classes: number of classes of the training data
		qc: Quality index of shape (n_lfs,) of type numpy.ndarray OR a float. Values must be between 0 and 1. Default is 0.85

	Return:
		
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

	assert m.shape[0] > int(n_sup)

	y_train_pred = predict_gm(params, params, m, s, k, n_classes, continuous_mask, qc_temp)
	kernel = get_similarity_kernel(y_train_pred)
	similarity = euclidean_distances(x_train)
	sim_mat = kernel * similarity
	fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state = 0, metric = 'precomputed', n_samples = n_sup)
	sim_sub = fl.fit_transform(sim_mat)
	indices = find_indices(sim_mat, sim_sub)

	x_L = data[0][indices]
	l_L = data[1][indices]
	m_L = data[2][indices]
	L_L = data[3][indices]
	d_L = data[4][indices]
	r_L = data[5][indices]
	s_L = data[6][indices]
	n_L = data[7][indices]
	k_L = data[8][indices]

	false_mask = np.ones(data[0].shape[0], dtype = bool)
	false_mask[indices] = False

	x_U = data[0][false_mask]
	l_U = data[1][false_mask]
	m_U = data[2][false_mask]
	L_U = data[3][false_mask]
	d_U = data[4][false_mask]
	r_U = data[5][false_mask]
	s_U = data[6][false_mask]
	n_U = data[7][false_mask]
	k_U = data[8][false_mask]

	pickle.dump(x_L, path_save_L)
	pickle.dump(l_L, path_save_L)
	pickle.dump(m_L, path_save_L)
	pickle.dump(L_L, path_save_L)
	pickle.dump(d_L, path_save_L)
	pickle.dump(r_L, path_save_L)
	pickle.dump(s_L, path_save_L)
	pickle.dump(n_L, path_save_L)
	pickle.dump(k_L, path_save_L)

	pickle.dump(x_U, path_save_U)
	pickle.dump(l_U, path_save_U)
	pickle.dump(m_U, path_save_U)
	pickle.dump(L_U, path_save_U)
	pickle.dump(d_U, path_save_U)
	pickle.dump(r_U, path_save_U)
	pickle.dump(s_U, path_save_U)
	pickle.dump(n_U, path_save_U)
	pickle.dump(k_U, path_save_U)

	return