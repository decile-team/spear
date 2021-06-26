import torch.nn as nn
from torch import log
import numpy as np

from .utils_cage import probability


def log_likelihood_loss_supervised(theta, pi, y, m, s, k, n_classes, continuous_mask, qc, device):
	'''
		Joint Learning utils: Negative log likelihood loss, used in loss 4 in :cite:p:`DBLP:journals/corr/abs-2008-09887`

	Args:
		theta: [n_classes, n_lfs], the parameters
		pi: [n_classes, n_lfs], the parameters
		m: [n_instances, n_lfs], m[i][j] is 1 if jth LF is triggered on ith instance, else it is 0
		s: [n_instances, n_lfs], s[i][j] is the continuous score of ith instance given by jth continuous LF
		k: [n_lfs], k[i] is the class of ith LF, range: 0 to num_classes-1
		n_classes: num of classes/labels
		continuous_mask: [n_lfs], continuous_mask[i] is 1 if ith LF has continuous counter part, else it is 0
		qc: a float value OR [n_lfs], qc[i] quality index for ith LF
		device: 'cuda' if drivers are available, else 'cpu'
	
	Return:
		a real value, summation over (the log of probability for an instance)
	'''
	prob = probability(theta, pi, m, s, k, n_classes, continuous_mask, qc, device)
	prob = (prob.t() / prob.sum(1)).t()
	return nn.NLLLoss()(log(prob), y)

def entropy(probabilities):
	'''
		Joint Learning utils: Entropy, Used in loss 2 in :cite:p:`DBLP:journals/corr/abs-2008-09887`

	Args:
		probabilities: [num_unsup_instances, num_classes], probabilities[i][j] is probability of ith instance being jth class
	
	Return:
		a real value, the entropy value of given probability
	'''
	entropy = - (probabilities * log(probabilities)).sum(1)
	return entropy.sum() / probabilities.shape[0]

def kl_divergence(probs_p, probs_q):
	'''
		Joint Learning utils: KL divergence of two probabilities, used in loss 6 in :cite:p:`DBLP:journals/corr/abs-2008-09887`
		
	Args:
		probs_p: [num_instances, num_classes]
		probs_q: [num_instances, num_classes]
	
	Return:
		a real value, the KL divergence of given probabilities
	'''
	return (probs_p * log(probs_p / probs_q)).sum() / probs_p.shape[0]


def find_indices(data, data_sub):
	'''
		A helper function for subset selection

	Args:
		data: the complete data, torch tensor of shape [num_instances, num_classes]
		data_sub: the subset of 'data' whose indices are to be found. Should be of same shape as 'data'
	
	Return:
		list of indices, to be found from the result of apricot library
	'''
	indices = []
	for element in data_sub:
		x = np.where((data.cpu().numpy() == element.cpu().numpy()).all(axis=1))[0]
		indices.append(x[0])
	return indices


def get_similarity_kernel(preds):
	'''
		A helper function for subset selection

	Args:
		preds: numpy.ndarray of shape (num_samples,)
	
	Return:
		numpy.ndarray of shape (num_sample, num_samples)

	'''
	num_samples = len(preds)
	kernel_matrix = np.zeros((num_samples, num_samples))
	for pred in np.unique(preds):
		x = np.where(preds == pred)[0]
		prod = np.transpose([np.tile(x, len(x)), np.repeat(x, len(x))])
		kernel_matrix[prod] = 1
	return kernel_matrix
