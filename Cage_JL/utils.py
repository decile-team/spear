import pickle
import numpy as np 
import torch
from torch.distributions.beta import Beta


def get_data(path):
	'''
		expected order in pickle file is NUMPY ndarrays x, l, m, L, d, r, s, n, k
			x: (num_instances, num_features), x[i][j] is jth feature of ith instance
			
			l: (num_instances, num_rules), l[i][j] is the prediction of jth LF(range: 0 to num_classes-1) on ith instance. l[i][j] = num_classes imply Abstain
			
			m: (num_instances, num_rules), m[i][j] is 1 if jth LF didn't Abstain on ith instance. Else it is 0
			
			L: (num_instances, 1), L[i] is true label(range: 0 to num_classes-1) of ith instance, if available. Else it is num_classes
			
			d: (num_instances, 1), d[i] is 1 if ith instance is labelled. Else it is 0
			
			r: (num_instances, num_rules), r[i][j] is 1 if ith instance is an exemplar for jth rule. Else it is 0
			
			s: (num_instances, num_rules), s[i][j] is the continuous score of ith instance given by jth continuous LF
			
			n: (num_rules,), n[i] is 1 if ith LF has continuous counter part, else it is 0
			
			k: (num_rules,), k[i] is the class of ith LF, range: 0 to num_classes-1

	Args: 
		path: path to pickle file with data in the format above

	Return:
		A list containing all the numpy arrays mentioned above
	'''
	data = []
	with open(path, 'rb') as file:
		for i in range(9):
			if i == 0:
				data.append(pickle.load(f))
			elif i == 6:
				data.append(pickle.load(f).astype(np.float32))
			else:
				data.append(pickle.load(f).astype(np.int32))

			assert type(data[i]) == np.ndarray

	assert data[1].shape == data[2].shape # l, m
	assert data[1].shape == data[5].shape # l, r
	assert data[1].shape == data[6].shape # l, s
	assert data[3].shape == (data[1].shape[0],1) #L, l
	assert data[4].shape == (data[1].shape[0],1) #d, l
	assert data[7].shape == (data[1].shape[1],) #n, l
	assert data[8].shape == (data[1].shape[1],) #k, l
	assert data[1].shape[0] == data[0].shape[0] #x, l
	assert np.all(np.logical_or(data[2] == 0, data[2] == 1)) #m
	assert np.all(np.logical_or(data[4] == 0, data[4] == 1)) #d
	assert np.all(np.logical_or(data[5] == 0, data[5] == 1)) #r
	assert np.all(np.logical_or(data[7] == 0, data[7] == 1)) #n

	return data

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
		Graphical model utils: Used to find Z(the normaliser) in CAGE

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


def probability_l_y(theta, l, k, n_classes):
	'''
		Graphical model utils: Used to find probability involving the term psi_theta, the potential function for all LFs

	Args:
		theta: [n_classes, n_lfs], the parameters
		l: [n_instances, n_lfs], l[i][j] is 1 if jth LF is triggered on ith instance, else it is 0
		k: [n_lfs], k[i] is the class of ith LF, range: 0 to num_classes-1
		n_classes: num of classes/labels
	Return:
		[n_instances, n_classes], the psi_theta value for each instance, for each class(true label y)
	'''
	probability = torch.zeros((l.shape[0], n_classes))
	z = calculate_normalizer(theta, k, n_classes)
	for y in range(n_classes):
		probability[:, y] = torch.exp(phi(theta[y], l).sum(1)) / z

	return probability.double()


def probability_s_given_y_l(pi, s, y, l, k, continuous_mask, qc):
	'''
		Graphical model utils: Used to find probability involving the term psi_pi, the potential function for all continuous LFs

	Args:
		s: [n_instances, n_lfs], s[i][j] is the continuous score of ith instance given by jth continuous LF
		y: a value in [0, n_classes-1], representing true label, for which psi_pi is calculated
		l: [n_instances, n_lfs], l[i][j] is 1 if jth LF is triggered on ith instance, else it is 0
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
		m = Beta(r[i] * params[i], params[i] * (1 - r[i]))
		probability *= torch.exp(m.log_prob(s[:, i].double())) * l[:, i].double() * continuous_mask[i] \
		+ (1 - l[:, i]).double() + (1 - continuous_mask[i])
	return probability


def probability(theta, pi, l, s, k, n_classes, continuous_mask, qc):
	'''
		Graphical model utils: Used to find probability of given instances for all possible y's

	Args:
		theta: [n_classes, n_lfs], the parameters
		pi: [n_classes, n_lfs], the parameters
		l: [n_instances, n_lfs], l[i][j] is 1 if jth LF is triggered on ith instance, else it is 0
		s: [n_instances, n_lfs], s[i][j] is the continuous score of ith instance given by jth continuous LF
		k: [n_lfs], k[i] is the class of ith LF, range: 0 to num_classes-1
		n_classes: num of classes/labels
		continuous_mask: [n_lfs], continuous_mask[i] is 1 if ith LF has continuous counter part, else it is 0
		qc: a float value OR [n_lfs], qc[i] quality index for ith LF. Value(s) must be between 0 and 1
	Return:
		[n_instances, n_classes], the probability for an instance being a particular class
	'''
	p_l_y = probability_l_y(theta, l, k, n_classes)
	p_s = torch.ones(s.shape[0], n_classes).double()
	for y in range(n_classes):
		p_s[:, y] = probability_s_given_y_l(pi[y], s, y, l, k, continuous_mask, qc)
	return p_l_y * p_s


def log_likelihood_loss(theta, pi, l, s, k, n_classes, continuous_mask, qc):
	'''
		Graphical model utils: negative of log likelihood loss

	Args:
		theta: [n_classes, n_lfs], the parameters
		pi: [n_classes, n_lfs], the parameters
		l: [n_instances, n_lfs], l[i][j] is 1 if jth LF is triggered on ith instance, else it is 0
		s: [n_instances, n_lfs], s[i][j] is the continuous score of ith instance given by jth continuous LF
		k: [n_lfs], k[i] is the class of ith LF, range: 0 to num_classes-1
		n_classes: num of classes/labels
		continuous_mask: [n_lfs], continuous_mask[i] is 1 if ith LF has continuous counter part, else it is 0
		qc: a float value OR [n_lfs], qc[i] quality index for ith LF. Value(s) must be between 0 and 1
	Return:
		a real value, summation over (the log of probability for an instance, marginalised over y(true labels))
	'''
	eps = 1e-8
	return - torch.log(probability(theta, pi, l, s, k, n_classes, continuous_mask, qc).sum(1) + eps).sum() / s.shape[0]


def precision_loss(theta, k, n_classes, a): 
	'''
		Graphical model utils: Precison loss, the R(theta) term in CAGE loss function

	Args:
		theta: [n_classes, n_lfs], the parameters
		k: [n_lfs], k[i] is the class of ith LF, range: 0 to num_classes-1
		n_classes: num of classes/labels
		a: [n_lfs], a[i] is the quality guide for ith LF. Value(s) must be between 0 and 1
	Return:
		a real value, R(t) term
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

def predict_gm(theta, pi, l, s, k, n_classes, continuous_mask, qc):
	'''
		Graphical model utils: Used to predict the labels after the training is done

	Args:
		theta: [n_classes, n_lfs], the parameters
		pi: [n_classes, n_lfs], the parameters
		l: [n_instances, n_lfs], l[i][j] is 1 if jth LF is triggered on ith instance, else it is 0
		s: [n_instances, n_lfs], s[i][j] is the continuous score of ith instance given by jth continuous LF
		k: [n_lfs], k[i] is the class of ith LF, range: 0 to num_classes-1
		n_classes: num of classes/labels
		continuous_mask: [n_lfs], continuous_mask[i] is 1 if ith LF has continuous counter part, else it is 0
		qc: a float value OR [n_lfs], qc[i] quality index for ith LF. Value(s) must be between 0 and 1
	Return:
		[n_instances], the predicted class for an instance
	'''
	return np.argmax(probability(theta, pi, l, s, k, n_classes, continuous_mask, qc).detach().numpy(), 1)