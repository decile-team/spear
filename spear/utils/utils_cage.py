'''
The common utils to CAGE and JL algorithms are in this file. Don't change the name or location of this file.
'''

import numpy as np 
import torch
from torch.distributions.beta import Beta

def phi(theta, l, device):
	'''
		Graphical model utils: A helper function

	Args:
		theta: [n_classes, n_lfs], the parameters
		l: [n_lfs]
		device: 'cuda' if drivers are available, else 'cpu'

	Return:
		a tensor of shape [n_classes, n_lfs], element wise product of input tensors(each row of theta dot product with l)
	'''
	return theta * torch.abs(l).double().to(device=device)


def calculate_normalizer(theta, k, n_classes, device):
	'''
		Graphical model utils: Used to find Z(the normaliser) in CAGE. Eq(4) in :cite:p:`2020:CAGE`

	Args:
		theta: [n_classes, n_lfs], the parameters
		k: [n_lfs], labels corresponding to LFs
		n_classes: num of classes/labels
		device: 'cuda' if drivers are available, else 'cpu'
	
	Return:
		a real value, representing the normaliser
	'''
	z = 0
	for y in range(n_classes):
		m_y = torch.exp(phi(theta[y], torch.ones(k.shape), device))
		z += (1 + m_y).prod()
	return z


def probability_l_y(theta, m, k, n_classes, device):
	'''
		Graphical model utils: Used to find probability involving the term psi_theta(in Eq(1) in :cite:p:`2020:CAGE`), the potential function for all LFs

	Args:
		theta: [n_classes, n_lfs], the parameters
		m: [n_instances, n_lfs], m[i][j] is 1 if jth LF is triggered on ith instance, else it is 0
		k: [n_lfs], k[i] is the class of ith LF, range: 0 to num_classes-1
		n_classes: num of classes/labels
		device: 'cuda' if drivers are available, else 'cpu'
	
	Return:
		a tensor of shape [n_instances, n_classes], the psi_theta value for each instance, for each class(true label y)
	'''
	probability = torch.zeros((m.shape[0], n_classes), device = device)
	z = calculate_normalizer(theta, k, n_classes, device)
	for y in range(n_classes):
		probability[:, y] = torch.exp(phi(theta[y], m, device).sum(1)) / z

	return probability.double()


def probability_s_given_y_l(pi, s, y, m, k, continuous_mask, qc):
	'''
		Graphical model utils: Used to find probability involving the term psi_pi(in Eq(1) in :cite:p:`2020:CAGE`), the potential function for all continuous LFs

	Args:
		pi: [n_lfs], the parameters for the class y
		s: [n_instances, n_lfs], s[i][j] is the continuous score of ith instance given by jth continuous LF
		y: a value in [0, n_classes-1], representing true label, for which psi_pi is calculated
		m: [n_instances, n_lfs], m[i][j] is 1 if jth LF is triggered on ith instance, else it is 0
		k: [n_lfs], k[i] is the class of ith LF, range: 0 to num_classes-1
		continuous_mask: [n_lfs], continuous_mask[i] is 1 if ith LF has continuous counter part, else it is 0
		qc: a float value OR [n_lfs], qc[i] quality index for ith LF. Value(s) must be between 0 and 1
	
	Return:
		a tensor of shape [n_instances], the psi_pi value for each instance, for the given label(true label y)
	'''
	eq = torch.eq(k.view(-1, 1), y).double().t()
	r = qc * eq.squeeze() + (1 - qc) * (1 - eq.squeeze())
	params = torch.exp(pi)
	probability = 1
	for i in range(k.shape[0]):
		temp = Beta(r[i] * params[i], params[i] * (1 - r[i]))
		probability *=  torch.exp(temp.log_prob(s[:, i].double())) * m[:, i].double() * continuous_mask[i] \
				 + (1 - m[:, i]).double() + (1 - continuous_mask[i])
	return probability


def probability(theta, pi, m, s, k, n_classes, continuous_mask, qc, device):
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
		device: 'cuda' if drivers are available, else 'cpu'
	
	Return:
		a tensor of shape [n_instances, n_classes], the probability for an instance being a particular class
	'''
	p_l_y = probability_l_y(theta, m, k, n_classes, device)
	p_s = torch.ones(s.shape[0], n_classes, device = device).double()
	for y in range(n_classes):
		p_s[:, y] = probability_s_given_y_l(pi[y], s, y, m, k, continuous_mask, qc)
	return p_l_y * p_s


def log_likelihood_loss(theta, pi, m, s, k, n_classes, continuous_mask, qc, device):
	'''
		Graphical model utils: Negative of log likelihood loss. Negative of Eq(6) in :cite:p:`2020:CAGE`

	Args:
		theta: [n_classes, n_lfs], the parameters
		pi: [n_classes, n_lfs], the parameters
		m: [n_instances, n_lfs], m[i][j] is 1 if jth LF is triggered on ith instance, else it is 0
		s: [n_instances, n_lfs], s[i][j] is the continuous score of ith instance given by jth continuous LF
		k: [n_lfs], k[i] is the class of ith LF, range: 0 to num_classes-1
		n_classes: num of classes/labels
		continuous_mask: [n_lfs], continuous_mask[i] is 1 if ith LF has continuous counter part, else it is 0
		qc: a float value OR [n_lfs], qc[i] quality index for ith LF. Value(s) must be between 0 and 1
		device: 'cuda' if drivers are available, else 'cpu'
	
	Return:
		a real value, negative of summation over (the log of probability for an instance, marginalised over y(true labels))
	'''
	eps = 1e-8
	return - torch.log(probability(theta, pi, m, s, k, n_classes, continuous_mask, qc, device).sum(1) + eps).sum() / s.shape[0]


def precision_loss(theta, k, n_classes, a, device): 
	'''
		Graphical model utils: Negative of the regularizer term in Eq(9) in :cite:p:`2020:CAGE`

	Args:
		theta: [n_classes, n_lfs], the parameters
		k: [n_lfs], k[i] is the class of ith LF, range: 0 to num_classes-1
		n_classes: num of classes/labels
		a: [n_lfs], a[i] is the quality guide for ith LF. Value(s) must be between 0 and 1
		device: 'cuda' if drivers are available, else 'cpu'
	
	Return:
		a real value, negative of regularizer term
	'''
	n_lfs = k.shape[0]
	prob = torch.ones(n_lfs, n_classes, device = device).double()
	z_per_lf = 0
	for y in range(n_classes):
		m_y = torch.exp(phi(theta[y], torch.ones(n_lfs, device = device), device))
		per_lf_matrix = torch.tensordot((1 + m_y).view(-1, 1), torch.ones(m_y.shape, device = device).double().view(1, -1), 1) \
		- torch.eye(n_lfs, device = device).double()
		prob[:, y] = per_lf_matrix.prod(0).double()
		z_per_lf += prob[:, y].double()
	prob /= z_per_lf.view(-1, 1)
	correct_prob = torch.zeros(n_lfs, device = device)
	for i in range(n_lfs):
		correct_prob[i] = prob[i, k[i]]
	loss = a * torch.log(correct_prob).double() + (1 - a) * torch.log(1 - correct_prob).double()
	return -loss.sum()

def predict_gm_labels(theta, pi, m, s, k, n_classes, continuous_mask, qc, device):
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
		device: 'cuda' if drivers are available, else 'cpu'
	
	Return:
		numpy.ndarray of shape (n_instances,), the predicted class for an instance
	'''
	return np.argmax(probability(theta, pi, m, s, k, n_classes, continuous_mask, qc, device).cpu().detach().numpy(), 1)