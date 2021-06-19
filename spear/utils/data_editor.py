'''
The common utils to CAGE and JL algorithms are in this file. Don't change the name or location of this file.
'''

import pickle, json
import numpy as np 
from os import path as check_path

def is_dict_trivial(dict):
	'''	
		A helper function that checks if the dictionary have key and value equal values for all keys except if its null

	Args:
		dict: the dictionary
	
	Return:
		True if all keys(which are not None) are equal to respective values. False otherwise
	'''
	for key, value in dict.items():
		if not(key == None):
			try:
				if key == value:
					continue
				else:
					return False
			except:
				return False
	return True


def get_data(path, check_shapes = True, class_map = None):
	'''
		Standard format in pickle file contains the NUMPY ndarrays x, l, m, L, d, r, s, n, k and an int n_classes
			x: (num_instances, num_features), x[i][j] is jth feature of ith instance. Note that the dimension fo this array can vary depending on the dimension of input
			
			l: (num_instances, num_lfs), l[i][j] is the prediction of jth LF(co-domain: the values used in Enum) on ith instance. l[i][j] = None imply Abstain
			
			m: (num_instances, num_lfs), m[i][j] is 1 if jth LF didn't Abstain on ith instance. Else it's 0
			
			L: (num_instances, 1), L[i] is true label(co-domain: the values used in Enum) of ith instance, if available. Else L[i] is None
			
			d: (num_instances, 1), d[i] is 1 if ith instance is labelled. Else it is 0
			
			r: (num_instances, num_lfs), r[i][j] is 1 if ith instance is an exemplar for jth rule. Else it's 0
			
			s: (num_instances, num_lfs), s[i][j] is the continuous score of ith instance given by jth continuous LF. If jth LF is not continuous, then s[i][j] is None
			
			n: (num_lfs,), n[i] is 1 if ith LF has continuous counter part, else n[i] is 0
			
			k: (num_lfs,), k[i] is the class of ith LF, co-domain: the values used in Enum

			n_classes: total number of classes

			In case the numpy array is not available(can be possible for x, L, d, r, s), it is stored as numpy.zeros(0)

	Args: 
		path: path to pickle file with data in the format above
		check_shapes: if true, checks whether the shapes of numpy arrays in pickle file are consistent as per the format mentioned above. Else it doesn't check. Default is True. 
		class_map: dictionary of class numbers(sorted, mapped to [0,n_classes-1]) are per the Enum defined in labeling part. l,L are modified(needed inside algorithms) before returning, using class_map. Default is None which doesn't do any mapping

	Return:
		A list containing all the numpy arrays mentioned above. The arrays l, L are modified using the class_map 
	'''
	assert check_path.exists(path)
	data = []
	with open(path, 'rb') as file:
		for i in range(9):
			data.append(pickle.load(file))
			assert type(data[i]) == np.ndarray
		data.append(pickle.load(file))
	
	assert type(data[9]) == np.int

	if check_shapes:
		assert data[1].shape == data[2].shape # l, m
		assert (data[1].shape == data[5].shape) or (data[5].shape[0] == 0) # l, r
		assert (data[1].shape == data[6].shape) or (data[6].shape[0] == 0) # l, s
		assert (data[3].shape == (data[1].shape[0],1)) or (data[3].shape[0] == 0) #L, l
		assert (data[4].shape == (data[1].shape[0],1)) or (data[4].shape[0] == 0) #d, l
		assert data[7].shape == (data[1].shape[1],) #n, l
		assert data[8].shape == (data[1].shape[1],) #k, l
		assert (data[0].shape[0] == 0) or data[1].shape[0] == data[0].shape[0] #x, l
		assert np.all(np.logical_or(data[2] == 0, data[2] == 1)) #m
		assert (data[4].shape[0] == 0) or (np.all(np.logical_or(data[4] == 0, data[4] == 1))) #d
		assert (data[5].shape[0] == 0) or (np.all(np.logical_or(data[5] == 0, data[5] == 1)) )#r
		assert np.all(np.logical_or(data[7] == 0, data[7] == 1)) #n

	if class_map == None:
		return data

	is_dict_trivial_ = is_dict_trivial(class_map)
	if not(is_dict_trivial_):
		data[1] = np.vectorize(class_map.get)(data[1])
		if not(data[3].shape[0] == 0):
			data[3] = np.vectorize(class_map.get)(data[3])
	else:
		data[1][data[1] == None] = data[9]
		if not(data[3].shape[0] == 0):
			data[3][data[3] == None] = data[9]

	data[6][data[6] == None] = 0 # s will have None values if LF is not continuous
	for i in range(9):
		if i == 0 or data[i].shape == 0:
			continue
		elif i == 6:
			data[i] = data[i].astype(np.float32)
		else:
			data[i] = data[i].astype(np.int32)

	return data

def get_classes(path):
	'''
		The json file should contain a dictionary of number to string(class name) map as defined in Enum

		Args:
			path: path to json file with contents mentioned above
		
		Returns:
			A dictionary (number to string(class name) map)
	'''
	assert check_path.exists(path)
	json_object = None
	with open(path, 'r') as f:
		json_object = json.load(f)
	json_object = {int(index): value for index, value in json_object.items()}
	return json_object

def get_predictions(proba, class_map, class_dict, need_strings):
	'''
		This function takes probaility of instances being a class and gives what class each instance belongs to, using the maximum of probabilities

	Args:
		proba: probability numpy.ndarray of shape (num_instances, num_classes)
		class_map: dictionary mapping the class numbers(as per Enum class defined) to numbers in range [0, num_classes-1]
		class_dict: dictionary consisting of number to string(class name) mapping as per the Enum class defined
		need_trings: If True, the output conatians strings(of class names), else it consists of numbers(class numbers as used in Enum definition)

	Return:
		numpy.ndarray of shape (num_instances,), where elements are class_names/class_numbers depending on need_strings is True/False, where the elements
		represent the class of each instance
	'''
	final_labels = np.argmax(proba, 1) # this is actually labels_with_altered_class_values
	if not(is_dict_trivial(class_map)):
		remap_dict = {value:index for index, value in (class_map).items()}
		final_labels = np.vectorize(remap_dict.get)(final_labels)
	if need_strings:
		class_dict_with_abstain = (class_dict).copy()
		class_dict_with_abstain[None] = 'ABSTAIN'
		return np.vectorize(class_dict_with_abstain.get)(final_labels)
	else:
		return final_labels

def get_enum(np_array, enm):
	'''
		This function is used to convert a numpy array of numbers to a numpy array of enums based on the Enum class provided 'enm'

	Args:
		np_array: a numpy.ndarray of any shape consisting of numbers
		enm: An class derived from 'Enum' class, which must contain map from every number in np_array to an enum

	Return:
		numpy.ndarray of shape shape as np_array but now contains enums(as per the mapping in 'enm') instead of numbers
	'''
	try:
		ans = np.vectorize(enm)(np_array)
	except:
		print("Error in get_enum function in utils.py: maybe enm doesn't containt the map for the numbers in np_array")
		exit(1)

	return ans
