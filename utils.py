'''
The common utils to CAGE and JL algorithms are in this file. Don't change the name or location of this file.
'''

import pickle, json
import numpy as np 
import torch
import os
from torch.distributions.beta import Beta


def get_data(path, class_map, check_shapes = True):
	'''
		Standard format in pickle file contains the NUMPY ndarrays x, l, m, L, d, r, s, n, k
			x: (num_instances, num_features), x[i][j] is jth feature of ith instance. Note that the dimension fo this array can vary depending on the dimension of input
			
			l: (num_instances, num_lfs), l[i][j] is the prediction of jth LF(co-domain: the values used in Enum) on ith instance. l[i][j] = None imply Abstain
			
			m: (num_instances, num_lfs), m[i][j] is 1 if jth LF didn't Abstain on ith instance. Else it's 0
			
			L: (num_instances, 1), L[i] is true label(co-domain: the values used in Enum) of ith instance, if available. Else L[i] is None
			
			d: (num_instances, 1), d[i] is 1 if ith instance is labelled. Else it is 0
			
			r: (num_instances, num_lfs), r[i][j] is 1 if ith instance is an exemplar for jth rule. Else it's 0
			
			s: (num_instances, num_lfs), s[i][j] is the continuous score of ith instance given by jth continuous LF
			
			n: (num_lfs,), n[i] is 1 if ith LF has continuous counter part, else n[i] is 0
			
			k: (num_lfs,), k[i] is the class of ith LF, co-domain: the values used in Enum

			n_classes: total number of classes

	Args: 
		path: path to pickle file with data in the format above
		class_map: dictionary of class numbers(sorted, mapped to [0,n_classes-1]) are per the Enum defined in labeling part
		check_shapes: if true, checks whether the shapes of numpy arrays in pickle file are consistent as per the format mentioned above. Else it doesn't check. Default is True. 

	Return:
		A list containing all the numpy arrays mentioned above. The arrays l, L are modified using the class_map 
	'''
	assert os.path.exists(path)
	data = []
	with open(path, 'rb') as file:
		for i in range(9):
			if i == 0:
				data.append(pickle.load(file))
			elif i == 6:
				data.append(pickle.load(file).astype(np.float32))
			else:
				data.append(pickle.load(file).astype(np.int32))

			assert type(data[i]) == np.ndarray
		data.append(pickle.load(file))

	if check_shapes:
		assert data[1].shape == data[2].shape # l, m
		assert (data[1].shape == data[5].shape) or (data[5].shape[0] == 0) # l, r
		assert (data[1].shape == data[6].shape) or (data[6].shape[0] == 0) # l, s
		assert (data[3].shape == (data[1].shape[0],1)) or (data[3].shape[0] == 0) #L, l
		assert (data[4].shape == (data[1].shape[0],1)) or (data[4].shape[0] == 0) #d, l
		assert data[7].shape == (data[1].shape[1],) #n, l
		assert data[8].shape == (data[1].shape[1],) #k, l
		assert data[1].shape[0] == data[0].shape[0] #x, l
		assert np.all(np.logical_or(data[2] == 0, data[2] == 1)) #m
		assert (data[4].shape[0] == 0) or (np.all(np.logical_or(data[4] == 0, data[4] == 1))) #d
		assert (data[5].shape[0] == 0) or (np.all(np.logical_or(data[5] == 0, data[5] == 1)) )#r
		assert np.all(np.logical_or(data[7] == 0, data[7] == 1)) #n

	data[1] = np.vectorize(class_map.get)(data[1])
	data[3] = np.vectorize(class_map.get)(data[3])

	return data

def get_classes(path):
	'''
		The json file should contain a dictionary of number to string(class name) map

		Args:
			path: path of json file with contents mentioned above
		
		Returns:
			A dictionary (number to string(class name) map)
	'''
	assert os.path.exists(path)
	json_object = None
	with open(path, 'r') as f:
		json_object = json.load(f)
	return json_object

def get_predictions(proba, class_map, class_dict, need_strings):
	'''
		This function takes probaility of instances being a class and what class each instance belongs to, using the maximum of probabilities

	Args:
		proba: probability numpy.ndarray of shape (num_instances, num_classes)
		class_map: dictionary mapping the class numbers(as per Enum class defined) to numbers in range [0, num_classes-1]
		class_dict: dictionary consisting of number to string(class name) mapping as per the Enum class defined
		need_trings: If True, the output conatians strings(of class names), else it consists of numbers(class numbers as used in Enum definition)

	Return:
		numpy.ndarray of shape (num_instances,), where elements are class_names/class_numbers depending on need_strings is True/False, where the elements
		represent the class of each instance
	'''
	labels_with_altered_class_values = np.argmax(proba.detach().numpy(), 1)
	remap_dict = {value:index for index, value in (class_map).items()}
	final_labels = np.vectorize(remap_dict.get)(labels_with_altered_class_values)
	if need_strings:
		class_dict_with_abstain = (class_dict).copy()
		class_dict_with_abstain[None] = 'Abstain'
		return np.vectorise(class_dict_with_abstain.get)(final_labels)
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

def phi(theta, l):
	'''
		A helper function

	Args:
		theta: [n_classes, n_lfs], the parameters
		l: [n_lfs]
	
	Return:
		[n_classes, n_lfs], element wise product of input tensors(each row of theta dot product with l)
	'''
	return theta * torch.abs(l).double()


def calculate_normalizer(theta, k, n_classes):
	'''
		Graphical model utils: Used to find Z(the normaliser) in CAGE. Eq(4) in :cite:p:`2020:CAGE`

	Args:
		theta: [n_classes, n_lfs], the parameters
		k: [n_lfs], labels corresponding to LFs
		n_classes: num of classes/labels
	
	Return:
		a real value, representing the normaliser
	'''
	z = 0
	for y in range(n_classes):
		m_y = torch.exp(phi(theta[y], torch.ones(k.shape)))
		z += (1 + m_y).prod()
	return z


def probability_l_y(theta, m, k, n_classes):
	'''
		Graphical model utils: Used to find probability involving the term psi_theta(in Eq(1) in :cite:p:`2020:CAGE`), the potential function for all LFs

	Args:
		theta: [n_classes, n_lfs], the parameters
		m: [n_instances, n_lfs], m[i][j] is 1 if jth LF is triggered on ith instance, else it is 0
		k: [n_lfs], k[i] is the class of ith LF, range: 0 to num_classes-1
		n_classes: num of classes/labels
	
	Return:
		[n_instances, n_classes], the psi_theta value for each instance, for each class(true label y)
	'''
	probability = torch.zeros((m.shape[0], n_classes))
	z = calculate_normalizer(theta, k, n_classes)
	for y in range(n_classes):
		probability[:, y] = torch.exp(phi(theta[y], m).sum(1)) / z

	return probability.double()


def probability_s_given_y_l(pi, s, y, m, k, continuous_mask, qc):
	'''
		Graphical model utils: Used to find probability involving the term psi_pi(in Eq(1) in :cite:p:`2020:CAGE`), the potential function for all continuous LFs

	Args:
		s: [n_instances, n_lfs], s[i][j] is the continuous score of ith instance given by jth continuous LF
		y: a value in [0, n_classes-1], representing true label, for which psi_pi is calculated
		m: [n_instances, n_lfs], m[i][j] is 1 if jth LF is triggered on ith instance, else it is 0
		k: [n_lfs], k[i] is the class of ith LF, range: 0 to num_classes-1
		continuous_mask: [n_lfs], continuous_mask[i] is 1 if ith LF has continuous counter part, else it is 0
		qc: a float value OR [n_lfs], qc[i] quality index for ith LF. Value(s) must be between 0 and 1
	
	Return:
		[n_instances], the psi_pi value for each instance, for the given label(true label y)
	'''
	eq = torch.eq(k.view(-1, 1), y).double().t()
	r = qc * eq.squeeze() + (1 - qc) * (1 - eq.squeeze())
	params = torch.exp(pi)
	probability = 1
	for i in range(k.shape[0]):
		temp = Beta(r[i] * params[i], params[i] * (1 - r[i]))
		probability *= torch.exp(temp.log_prob(s[:, i].double())) * m[:, i].double() * continuous_mask[i] \
		+ (1 - m[:, i]).double() + (1 - continuous_mask[i])
	return probability


def probability(theta, pi, m, s, k, n_classes, continuous_mask, qc):
	'''
		Graphical model utils: Used to find probability of given instances for all possible true labels(y's). Eq(1) in :cite:p:`2020:CAGE`

	Args:
		theta: [n_classes, n_lfs], the parameters
		pi: [n_classes, n_lfs], the parameters
		m: [n_instances, n_lfs], m[i][j] is 1 if jth LF is triggered on ith instance, else it is 0
		s: [n_instances, n_lfs], s[i][j] is the continuous score of ith instance given by jth continuous LF
		k: [n_lfs], k[i] is the class of ith LF, range: 0 to num_classes-1
		n_classes: num of classes/labels
		continuous_mask: [n_lfs], continuous_mask[i] is 1 if ith LF has continuous counter part, else it is 0
		qc: a float value OR [n_lfs], qc[i] quality index for ith LF. Value(s) must be between 0 and 1
	
	Return:
		[n_instances, n_classes], the probability for an instance being a particular class
	'''
	p_l_y = probability_l_y(theta, m, k, n_classes)
	p_s = torch.ones(s.shape[0], n_classes).double()
	for y in range(n_classes):
		p_s[:, y] = probability_s_given_y_l(pi[y], s, y, m, k, continuous_mask, qc)
	return p_l_y * p_s


def log_likelihood_loss(theta, pi, m, s, k, n_classes, continuous_mask, qc):
	'''
		Graphical model utils: negative of log likelihood loss. Negative of Eq(6) in :cite:p:`2020:CAGE`

	Args:
		theta: [n_classes, n_lfs], the parameters
		pi: [n_classes, n_lfs], the parameters
		m: [n_instances, n_lfs], m[i][j] is 1 if jth LF is triggered on ith instance, else it is 0
		s: [n_instances, n_lfs], s[i][j] is the continuous score of ith instance given by jth continuous LF
		k: [n_lfs], k[i] is the class of ith LF, range: 0 to num_classes-1
		n_classes: num of classes/labels
		continuous_mask: [n_lfs], continuous_mask[i] is 1 if ith LF has continuous counter part, else it is 0
		qc: a float value OR [n_lfs], qc[i] quality index for ith LF. Value(s) must be between 0 and 1
	
	Return:
		a real value, summation over (the log of probability for an instance, marginalised over y(true labels))
	'''
	eps = 1e-8
	return - torch.log(probability(theta, pi, m, s, k, n_classes, continuous_mask, qc).sum(1) + eps).sum() / s.shape[0]


def precision_loss(theta, k, n_classes, a): 
	'''
		Graphical model utils: Negative of the regularizer term in Eq(9) in :cite:p:`2020:CAGE`

	Args:
		theta: [n_classes, n_lfs], the parameters
		k: [n_lfs], k[i] is the class of ith LF, range: 0 to num_classes-1
		n_classes: num of classes/labels
		a: [n_lfs], a[i] is the quality guide for ith LF. Value(s) must be between 0 and 1
	
	Return:
		a real value, negative of regularizer term
	'''
	n_lfs = k.shape[0]
	prob = torch.ones(n_lfs, n_classes).double()
	z_per_lf = 0
	for y in range(n_classes):
		m_y = torch.exp(phi(theta[y], torch.ones(n_lfs)))
		per_lf_matrix = torch.tensordot((1 + m_y).view(-1, 1), torch.ones(m_y.shape).double().view(1, -1), 1) \
		- torch.eye(n_lfs).double()
		prob[:, y] = per_lf_matrix.prod(0).double()
		z_per_lf += prob[:, y].double()
	prob /= z_per_lf.view(-1, 1)
	correct_prob = torch.zeros(n_lfs)
	for i in range(n_lfs):
		correct_prob[i] = prob[i, k[i]]
	loss = a * torch.log(correct_prob).double() + (1 - a) * torch.log(1 - correct_prob).double()
	return -loss.sum()

def predict_gm(theta, pi, m, s, k, n_classes, continuous_mask, qc):
	'''
		Graphical model utils: Used to predict the labels after the training is done

	Args:
		theta: [n_classes, n_lfs], the parameters
		pi: [n_classes, n_lfs], the parameters
		m: [n_instances, n_lfs], m[i][j] is 1 if jth LF is triggered on ith instance, else it is 0
		s: [n_instances, n_lfs], s[i][j] is the continuous score of ith instance given by jth continuous LF
		k: [n_lfs], k[i] is the class of ith LF, range: 0 to num_classes-1
		n_classes: num of classes/labels
		continuous_mask: [n_lfs], continuous_mask[i] is 1 if ith LF has continuous counter part, else it is 0
		qc: a float value OR [n_lfs], qc[i] quality index for ith LF. Value(s) must be between 0 and 1
	
	Return:
		numpy.ndarray of shape (n_instances,), the predicted class for an instance
	'''
	return np.argmax(probability(theta, pi, m, s, k, n_classes, continuous_mask, qc).detach().numpy(), 1)