from utils import *

import torch.nn as nn

def log_likelihood_loss_supervised(theta, pi_y, pi, y, l, s, k, n_classes, continuous_mask, qc):
	'''
		Joint Learning utils: Negative log likelihood loss used in loss 4

	Args:
		theta: [n_classes, n_lfs], the parameters
		pi: [n_classes, n_lfs], the parameters
		l: [n_instances, n_lfs], l[i][j] is 1 if jth LF is triggered on ith instance, else it is 0
		s: [n_instances, n_lfs], s[i][j] is the continuous score of ith instance given by jth continuous LF
		k: [n_lfs], k[i] is the class of ith LF, range: 0 to num_classes-1
		n_classes: num of classes/labels
		continuous_mask: [n_lfs], continuous_mask[i] is 1 if ith LF has continuous counter part, else it is 0
		qc: a float value OR [n_lfs], qc[i] quality index for ith LF
	Return:
		a real value, summation over (the log of probability for an instance)
	'''
	eps = 1e-8
	prob = probability(theta, pi_y, pi, l, s, k, n_classes, continuous_mask, qc)
	prob = (prob.t() / prob.sum(1)).t()
	return nn.NLLLoss()(torch.log(prob), y)

def entropy(probabilities):
	'''
		Joint Learning utils: Used in loss 2

	Args:
		probabilities: [num_unsup_instances, num_classes], probabilities[i][j] is probability of ith instance being jth class
	Return:
		a real value, the entropy value of given probability
	'''
	entropy = - (probabilities * torch.log(probabilities)).sum(1)
	return entropy.sum() / probabilities.shape[0]

def kl_divergence(probs_p, probs_q):
	'''
		Joint Learning utils: KL divergence of two probabilities, used in loss 6
		
	Args:
		probs_p: [num_instances, num_classes]
		probs_q: [num_instances, num_classes]
	Return:
		a real value, the KL divergence of given probabilities
	'''
	return (probs_p * torch.log(probs_p / probs_q)).sum() / probs_p.shape[0]