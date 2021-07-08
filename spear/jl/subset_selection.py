from submodlib.functions.facilityLocation import FacilityLocationFunction

import numpy as np
import torch
from sklearn.metrics.pairwise import euclidean_distances
import pickle
from os import path as check_path

from ..utils.data_editor import get_data, get_classes
from ..utils.utils_cage import  predict_gm_labels
from ..utils.utils_jl import find_indices, get_similarity_kernel

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
	return np.sort(np.random.choice(int(n_all), int(n_instances), replace = False))

def unsup_subset(x_train, n_unsup):
	'''
		A function for unsupervised subset selection(the subset to be labeled)
		
	Args:
		x_train: A numpy.ndarray of shape (n_instances, n_features). All the data, intended to be used for training
		n_unsup: number of instances to be found during unsupervised subset selection, type is integer
	
	Return:
		numpy.ndarray of indices(shape is (n_sup,), each element lies in [0,x_train.shape[0])), the result of subset selection
	'''
	assert x_train.shape[0] > int(n_unsup)
	assert type(x_train) == np.ndarray
	
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	torch.backends.cudnn.benchmark = True

	#fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state = 0, n_samples = int(n_unsup))
	#x_sub = fl.fit_transform(x_train)
	#indices = find_indices(torch.from_numpy(x_train).to(device=device), torch.from_numpy(np.array(x_sub)).to(device=device))

	fl = FacilityLocationFunction(n = x_train.shape[0], mode = "dense", data = x_train, metric = "euclidean")
	x_sub = fl.maximize(budget = int(n_unsup), optimizer = 'LazyGreedy', stopIfZeroGain = False, stopIfNegativeGain = False, verbose = False)
	indices = np.array([i[0] for i in x_sub])

	return np.sort(indices)

def sup_subset(path_json, path_pkl, n_sup, qc = 0.85):
	'''
		A helper function for supervised subset selection(the subset to be labeled) which just returns indices
	
	Args:
		path_json: Path to json file of number to string(class name) map
		path_pkl: Path to the pickle file containing all the training data in standard format
		n_sup: Number of instances to be found during supervised subset selection
		qc: Quality index of shape (n_lfs,) of type numpy.ndarray OR a float. Values must be between 0 and 1. Default is 0.85

	Return:
		numpy.ndarray of indices(shape is (n_sup,), each element lies in [0,num_instances)), the result of subset selection AND the data which is list of contents of path_pkl
	'''
	assert (type(qc) == np.float and (qc >= 0 and qc <= 1)) or (type(qc) == np.ndarray and (np.all(np.logical_and(qc>=0, qc<=1)) ) )\
		 or (type(qc) == np.int and (qc == 0 or qc == 1))

	class_dict = get_classes(path_json)
	class_list = list((class_dict).keys())
	class_list.sort()
	n_classes = len(class_dict)

	class_map = {value : index for index, value in enumerate(class_list)}
	class_map[None] = n_classes

	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	torch.backends.cudnn.benchmark = True

	data = get_data(path_pkl, True, class_map)
	m = torch.abs(torch.tensor(data[2], device = device).long())
	s = torch.tensor(data[6], device = device).double() # continuous score
	assert m.shape[0] > int(n_sup)
	k = torch.tensor(data[8], device = device).long() # LF's classes
	n_lfs = m.shape[1]
	continuous_mask = torch.tensor(data[7], device = device).double() # Mask for s/continuous_mask
	qc_temp = torch.tensor(qc, device = device).double() if type(qc) == np.ndarray else qc
	params_1 = torch.ones((n_classes, n_lfs), device = device).double() # initialisation of gm parameters, refer section 3.4 in the JL paper
	params_2 = torch.ones((n_classes, n_lfs), device = device).double()

	y_train_pred = predict_gm_labels(params_1, params_2, m, s, k, n_classes, continuous_mask, qc_temp, device)
	kernel = get_similarity_kernel(y_train_pred)
	similarity = euclidean_distances(data[0])
	sim_mat = kernel * similarity

	#fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state = 0, metric = 'precomputed', n_samples = int(n_sup))
	#sim_sub = fl.fit_transform(sim_mat)
	#indices = find_indices(torch.from_numpy(sim_mat).to(device=device), torch.from_numpy(np.array(sim_sub)).to(device=device))

	fl = FacilityLocationFunction(n = sim_mat.shape[0], mode = "dense", sijs = sim_mat, separate_rep = False)
	sim_sub = fl.maximize(budget = int(n_sup), optimizer = 'LazyGreedy', stopIfZeroGain = False, stopIfNegativeGain = False, verbose = False)
	indices = np.array([i[0] for i in sim_sub])

	return np.sort(indices), data

def sup_subset_indices(path_json, path_pkl, n_sup, qc = 0.85):
	'''
		A function for supervised subset selection(the subset to be labeled) whcih just returns indices
	
	Args:
		path_json: Path to json file of number to string(class name) map
		path_pkl: Path to the pickle file containing all the training data in standard format
		n_sup: Number of instances to be found during supervised subset selection
		qc: Quality index of shape (n_lfs,) of type numpy.ndarray OR a float. Values must be between 0 and 1. Default is 0.85

	Return:
		numpy.ndarray of indices(shape is (n_sup,), each element lies in [0,num_instances)), the result of subset selection
	'''
	indices, _ = sup_subset(path_json, path_pkl, n_sup, qc)

	return indices

def sup_subset_save_files(path_json, path_pkl, path_save_L, path_save_U, n_sup, qc = 0.85):
	'''
		A function for supervised subset selection(the subset to be labeled) which makes separate pickle files of data, one for those to be labelled, other that can be left unlabelled
	
	Args:
		path_json: Path to json file of number to string(class name) map
		path_pkl: Path to the pickle file containing all the training data in standard format
		path_save_L: Path to save the pickle file of set of instances to be labelled. Note that instances are not labelled yet. Extension should be .pkl
		path_save_U: Path to save the pickle file of set of instances that can be left unlabelled. Extension should be .pkl
		n_sup: number of instances to be found during supervised subset selection
		qc: Quality index of shape (n_lfs,) of type numpy.ndarray OR a float. Values must be between 0 and 1. Default is 0.85

	Return:
		numpy.ndarray of indices(shape is (n_sup,), each element lies in [0,num_instances)), the result of subset selection. Also two pickle files are saved at path_save_L and path_save_U
	'''
	indices, data = sup_subset(path_json, path_pkl, n_sup, qc)

	false_mask = np.ones(data[0].shape[0], dtype = bool)
	false_mask[indices] = False

	save_file_L = open(path_save_L, 'wb')
	save_file_U = open(path_save_U, 'wb')

	for i in range(10):
		if i < 7:
			if data[i].shape[0] == 0:
				pickle.dump(data[i], save_file_L)
				pickle.dump(data[i], save_file_U)
			else:
				pickle.dump(data[i][indices], save_file_L)
				pickle.dump(data[i][false_mask], save_file_U)
		elif i >= 7:
			pickle.dump(data[i], save_file_L)
			pickle.dump(data[i], save_file_U)

	save_file_L.close()
	save_file_U.close()

	return indices


def replace_in_pkl(path, path_save, np_array, index):
	'''
		A function to insert the true labels, after labeling the instances, to the pickle file

	Args:
		path: Path to the pickle file containing all the data in standard format
		path_save: Path to save the pickle file after replacing the 'L'(true labels numpy array) of data in path pickle file
		np_array: The data which is to be used to replace the data in path pickle file with
		index: Index of the numpy array, in data of path pickle file, to be replaced with np_array. Value should be in [0,8]
	Return:
		No return value. A pickle file is generated at path_save
	'''
	assert type(index) == np.int and index >=0 and index < 9
	assert check_path.exists(path) #path is imported from os above
	data = []
	with open(path, 'rb') as file:
		for i in range(9):
			data.append(pickle.load(file))
			assert type(data[i]) == np.ndarray
		data.append(pickle.load(file))

	assert data[index].shape[0] == 0 or np_array.shape == data[index].shape

	save_file = open(path_save, 'wb')
	for i in range(10):
		if i == index:
			pickle.dump(np_array, save_file)
		else:
			pickle.dump(data[i], save_file)
	save_file.close()

	return
	
def insert_true_labels(path, path_save, labels):
	'''
		A function to insert the true labels, after labeling the instances, to the pickle file

	Args:
		path: Path to the pickle file containing all the data in standard format
		path_save: Path to save the pickle file after replacing the 'L'(true labels numpy array) of data in path pickle file
		labels: The true labels of the data in pickle file. numpy.ndarray of shape (num_instances, 1)

	Return:
		No return value. A pickle file is generated at path_save
	'''
	assert check_path.exists(path) #path is imported from os above
	data = []
	with open(path, 'rb') as file:
		for i in range(9):
			data.append(pickle.load(file))
			assert type(data[i]) == np.ndarray
		data.append(pickle.load(file))

	assert labels.shape[0] == data[0].shape[0] and labels.shape[1] == 1

	save_file = open(path_save, 'wb')
	for i in range(10):
		if i == 3:
			pickle.dump(labels, save_file)
		elif i == 4:
			pickle.dump(np.ones([labels.size ,1]), save_file)
		else:
			pickle.dump(data[i], save_file)
	save_file.close()

	return