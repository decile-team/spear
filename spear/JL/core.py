import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from copy import deepcopy
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score as prec_score
from sklearn.metrics import recall_score as recall_score

from ..utils.data_editer import get_data, get_classes, get_predictions
from ..utils.utils_cage import probability, log_likelihood_loss, precision_loss, predict_gm_labels
from ..utils.utils_jl import log_likelihood_loss_supervised, entropy, kl_divergence
from .models.models import *


class JL:
	'''
	Joint_Learning class:
		[Note: from here on, feature model(fm) and feature-based classification model are used interchangeably. graphical model(gm) and CAGE algorithm terms are used interchangeably]
	
	Args:
		path_json: Path to json file containing the dictionary of number to string(class name) map
		path_L: Path to pickle file of labelled instances
		path_U: Path to pickle file of unlabelled instances
		path_V: Path to pickle file of validation instances
		path_T: Path to pickle file of test instances
	'''
	def __init__(self, path_json, path_L, path_U, path_V, path_T):

		assert type(path_json) == str and type(path_L) == str and type(path_V) == str and type(path_V) == str and type(path_T) == str
		torch.set_default_dtype(torch.float64)
		
		self.class_dict = get_classes(path_json)
		self.class_list = list((self.class_dict).keys())
		self.class_list.sort()
		self.n_classes = len(self.class_dict)

		self.class_map = {index : value for index, value in enumerate(self.class_list)}
		self.class_map[None] = self.n_classes

		data_L = get_data(path_L, True, self.class_map)
		data_U = get_data(path_U, True, self.class_map)
		data_V = get_data(path_V, True, self.class_map)
		data_T = get_data(path_T, True, self.class_map)

		self.x_sup = torch.tensor(data_L[0]).double()
		self.y_sup = torch.tensor(data_L[3]).long()
		self.l_sup = torch.tensor(data_L[2]).long()
		self.s_sup = torch.tensor(data_L[6]).double()

		excluding = []
		temp_index = 0
		for temp in data_U[1]:
			if(np.all(temp == int(self.n_classes)) ):
				excluding.append(temp_index)
			temp_index+=1

		self.x_unsup = torch.tensor(np.delete(data_U[0], excluding, axis=0)).double()
		self.y_unsup = torch.tensor(np.delete(data_U[3], excluding, axis=0)).long()
		self.l_unsup = torch.tensor(np.delete(data_U[2], excluding, axis=0)).long()
		self.s_unsup = torch.tensor(np.delete(data_U[6], excluding, axis=0)).double()

		self.x_valid = torch.tensor(data_V[0]).double()
		self.y_valid = data_V[3]
		self.l_valid = torch.tensor(data_V[2]).long()
		self.s_valid = torch.tensor(data_V[6]).double()

		self.x_test = torch.tensor(data_T[0]).double()
		self.y_test = data_T[3]
		self.l_test = torch.tensor(data_T[2]).long()
		self.s_test = torch.tensor(data_T[6]).double()

		self.y_unsup = (self.y_unsup).view(-1)
		self.y_sup = (self.y_sup).view(-1)
		self.y_valid = (self.y_valid).flatten()
		self.y_test = (self.y_test).flatten()

		self.k = torch.tensor(data_L[8]).long() # LF's classes
		self.n_lfs = self.l_sup.shape[1]
		self.continuous_mask = torch.tensor(data_L[7]).double() # Mask for s/continuous_mask

		assert np.all(data_L[8] == data_U[8]) and np.all(data_L[8] == data_V[8]) and np.all(data_L[8] == data_T[8])
		assert np.all(data_L[7] == data_U[7]) and np.all(data_L[7] == data_V[7]) and np.all(data_L[7] == data_T[7])

		#[Note: 
		#1. Loss function number, Calculated over, Loss function:
		#		1, L, Cross Entropy(prob_from_feature_model, true_labels)
		#		2, U, Entropy(prob_from_feature_model)
		#		3, U, Cross Entropy(prob_from_feature_model, prob_from_graphical_model)
		#		4, L, Negative Log Likelihood
		#		5, U, Negative Log Likelihood(marginalised over true labels)
		#		6, L and U, KL Divergence(prob_feature_model, prob_graphical_model)
		#		7, Quality guide
		#
		#2. each pickle file should follow the standard convention for data storage]
		#
		#3. shapes of x,y,l,s:
		#	x: [num_instances, num_features], feature matrix
		#	y: [num_instances, 1], true labels, if available
		#	l: [num_instances, num_rules], 1 if LF is triggered, 0 else
		#	s: [num_instances, num_rules], continuous score
		#]
		
		assert self.x_sup.shape[0] == self.y_sup.shape[0] and self.x_sup.shape[0] == self.l_sup.shape[0]\
		 and self.l_sup.shape == self.s_sup.shape and self.l_sup.shape[1] == self.n_lfs
		assert self.x_unsup.shape[0] == self.y_unsup.shape[0] and self.x_unsup.shape[0] == self.l_unsup.shape[0]\
		 and self.l_unsup.shape == self.s_unsup.shape and self.l_unsup.shape[1] == self.n_lfs
		assert self.x_valid.shape[0] == self.y_valid.shape[0] and self.x_valid.shape[0] == self.l_valid.shape[0]\
		 and self.l_valid.shape == self.s_valid.shape and self.l_valid.shape[1] == self.n_lfs
		assert self.x_test.shape[0] == self.y_test.shape[0] and self.x_test.shape[0] == self.l_test.shape[0]\
		 and self.l_test.shape == self.s_test.shape and self.l_test.shape[1] == self.n_lfs

		# clip s
		self.s_sup[self.s_sup > 0.999] = 0.999
		self.s_sup[self.s_sup < 0.001] = 0.001
		self.s_unsup[self.s_unsup > 0.999] = 0.999
		self.s_unsup[self.s_unsup < 0.001] = 0.001
		self.s_valid[self.s_valid > 0.999] = 0.999
		self.s_valid[self.s_valid < 0.001] = 0.001
		self.s_test[self.s_test > 0.999] = 0.999
		self.s_test[self.s_test < 0.001] = 0.001

		self.l = torch.cat([self.l_sup, self.l_unsup])
		self.s = torch.cat([self.s_sup, self.s_unsup])
		self.x_train = torch.cat([self.x_sup, self.x_unsup])
		self.y_train = torch.cat([self.y_sup, self.y_unsup])
		self.supervised_mask = torch.cat([torch.ones(self.l_sup.shape[0]), torch.zeros(self.l_unsup.shape[0])])

		##adding subsetselection here for testing. todo: has to be removed.
		#indices = rand_subset(self.x_train.shape[0], len(self.x_sup))
		#indices = unsup_subset(self.x_train, len(self.x_sup))
		#
		#self.supervised_mask = torch.zeros(self.x_train.shape[0])
		#self.supervised_mask[indices] = 1
		##

		self.pi, self.theta = None, None
		self.is_training_done = False

		self.pi_optimal, self.theta_optimal = None, None
		self.fm_optimal_params = None

	def fit_and_predict_proba(self, loss_func_mask, batch_size, lr_fm, lr_gm, use_accuracy_score, path_log = None, return_gm = False, n_epochs = 100, start_len = 7,\
	 stop_len = 10, is_qt = True, is_qc = True, qt = 0.9, qc = 0.85, n_hidden = 512, feature_model = 'nn', metric_avg = 'macro'):
		'''
		Args:
			loss_func_mask: list/numpy array of size 7 or (7,) where loss_func_mask[i] should be 1 if Loss function (i+1) should be included, 0 else. Checkout Eq(3) in :cite:p:`2020:JL`
			batch_size: Batch size used in each epoch during training, type should be integer
			lr_fm: Learning rate for feature model, type is integer or float
			lr_gm: Learning rate for graphical model(cage algorithm), type is integer or float
			use_accuracy_score: The score to use for termination condition on validation set. True for accuracy_score, False for f1_score
			path_log: Path to log file
			return_gm: Return the predictions of graphical model? the allowed values are True, False. Default value is False
			n_epochs: Number of epochs in each run, type is integer, default is 100
			start_len: A parameter used in validation refers to the least epoch after which validation checks need to be performed, type is integer, default is 7
			stop_len: A parameter used in validation refers to the least number of continuous epochs of non incresing validation accuracy after which the training should be stopped, type is integer, default is 10
			is_qt: True if quality guide is available(and will be provided in 'qt' argument). False if quality guide is intended to be found from validation instances. Default is True
			is_qc: True if quality index is available(and will be provided in 'qc' argument). False if quality index is intended to be found from validation instances. Default is True
			qt: Quality guide of shape (n_lfs,) of type numpy.ndarray OR a float. Values must be between 0 and 1. Default is 0.9
			qc: Quality index of shape (n_lfs,) of type numpy.ndarray OR a float. Values must be between 0 and 1. Default is 0.85
			n_hidden: Number of hidden layer nodes if feature model is 'nn', type is integer, default is 512
			feature_model: The model intended to be used for features, allowed values are 'lr'(Logistic Regression) or 'nn'(Neural network with 1 hidden layer) string, default is 'nn'
			metric_avg: Average metric to be used in calculating f1_score/precision/recall, default is 'macro'

		Return:
			If return_gm is True; the return value is two predicted labels of numpy array of shape (num_instances, num_classes), first one is through feature model, other one through graphical model.
			Else; the return value is predicted labels of numpy array of shape (num_instances, num_classes) through feature model. For a given model i,j-th element is the probability of ith instance being the 
			jth class(the jth value when sorted in ascending order of values in Enum) using that model
		'''

		assert type(return_gm) == np.bool
		assert len(loss_func_mask) == 7
		assert type(batch_size) == np.int or type(batch_size) == np.float
		assert type(lr_fm) == np.int or type(lr_fm) == np.float
		assert type(lr_gm) == np.int or type(lr_gm) == np.float
		assert type(use_accuracy_score) == np.bool
		assert type(n_epochs) == np.int or type(n_epochs) == np.float
		assert type(start_len) == np.int or type(start_len) == np.float
		assert type(stop_len) == np.int or type(stop_len) == np.float
		assert type(is_qt) == np.bool and type(is_qc) == np.bool
		assert (type(qt) == np.float and (qt >= 0 and qt <= 1)) or (type(qt) == np.ndarray and (np.all(np.logical_and(qt>=0, qt<=1)) ) )\
		 or (type(qt) == np.int and (qt == 0 or qt == 1))
		assert (type(qc) == np.float and (qc >= 0 and qc <= 1)) or (type(qc) == np.ndarray and (np.all(np.logical_and(qc>=0, qc<=1)) ) )\
		 or (type(qc) == np.int and (qc == 0 or qc == 1))
		assert type(n_hidden) == np.int or type(n_hidden) == np.float
		assert feature_model == 'lr' or feature_model == 'nn'
		assert type(metric_avg) == str


		self.loss_func_mask = loss_func_mask
		self.batch_size = int(batch_size)
		self.lr_fm = lr_fm
		self.lr_gm = lr_gm
		self.n_epochs = int(n_epochs)
		self.start_len = int(start_len)
		self.stop_len = int(stop_len)
		self.n_hidden = int(n_hidden)
		self.feature_based_model = feature_model
		self.metric_avg = metric_avg

		self.use_accuracy_score = use_accuracy_score
		self.score_used = "accuracy_score" if self.use_accuracy_score else "f1_score"

		assert self.start_len <= self.n_epochs and self.stop_len <= self.n_epochs

		if is_qt:
			self.qt = torch.tensor(qt).double() if type(qt) == np.ndarray else (torch.ones(self.n_lfs).double() * qt)
		else:
			prec_lfs=[]
			for i in range(self.n_lfs):
				correct = 0
				for j in range(len(self.y_valid)):
					if self.y_valid[j] == self.l_valid[j][i]:
						correct+=1
				prec_lfs.append(correct/len(self.y_valid))
			self.qt = torch.tensor(prec_lfs).double()

		if is_qc:
			self.qc = torch.tensor(qc).double() if type(qc) == np.ndarray else qc
		else:
			self.qc = torch.tensor(np.mean(self.s_valid, axis = 0))

		if self.feature_based_model == 'lr':
			self.feature_model = LogisticRegression(self.n_features, self.n_classes)
		elif self.feature_based_model =='nn':
			self.feature_model = DeepNet(self.n_features, self.n_hidden, self.n_classes)

		self.fm_optimal_params = deepcopy((self.feature_model).state_dict())

		file = None
		if path_log != None:
			file = open(path_log, "a+")
			file.write("JL log:\n")

		self.stopped_early = False

		#Algo starting

		self.pi = torch.ones((self.n_classes, self.n_lfs)).double()
		(self.pi).requires_grad = True

		self.theta = torch.ones((self.n_classes, self.n_lfs)).double()
		(self.theta).requires_grad = True

		self.pi_optimal, self.theta_optimal = (self.pi).detach().clone(), (self.theta).detach().clone()

		optimizer_fm = torch.optim.Adam(self.feature_model.parameters(), lr = self.lr_fm)
		optimizer_gm = torch.optim.Adam([self.theta, self.pi], lr = self.lr_gm, weight_decay=0)
		supervised_criterion = torch.nn.CrossEntropyLoss()

		dataset = TensorDataset(self.x_train, self.y_train, self.l, self.s, self.supervised_mask)
		loader = DataLoader(dataset, batch_size = self.batch_size, shuffle = True, pin_memory = True)

		best_score_fm_test, best_score_gm_test, best_epoch, best_score_fm_val, best_score_gm_val = 0,0,0,0,0
		best_prec_fm_test, best_recall_fm_test, best_prec_gm_test, best_recall_gm_test= 0,0,0,0

		gm_test_acc, fm_test_acc = -1, -1

		self.stopped_early = False
		stop_early_fm, stop_early_gm = [], []

		for epoch in range(self.n_epochs):
			
			self.feature_model.train()

			for _, sample in enumerate(loader):
				optimizer_fm.zero_grad()
				optimizer_gm.zero_grad()

				supervised_indices = sample[4].nonzero().view(-1)
				unsupervised_indices = (1-sample[4]).nonzero().squeeze()

				if(self.loss_func_mask[0]):
					if len(supervised_indices) > 0:
						loss_1 = supervised_criterion(self.feature_model(sample[0][supervised_indices]), sample[1][supervised_indices])
					else:
						loss_1 = 0
				else:
					loss_1=0

				if(self.loss_func_mask[1]):
					unsupervised_fm_probability = torch.nn.softmax(dim = 1)(self.feature_model(sample[0][unsupervised_indices]))
					loss_2 = entropy(unsupervised_fm_probability)
				else:
					loss_2=0

				if(self.loss_func_mask[2]):
					y_pred_unsupervised = predict_gm_labels(self.theta, self.pi, sample[2][unsupervised_indices], sample[3][unsupervised_indices], self.k, self.n_classes, self.continuous_mask, self.qc)
					loss_3 = supervised_criterion(self.feature_model(sample[0][unsupervised_indices]), torch.tensor(y_pred_unsupervised))
				else:
					loss_3 = 0

				if (self.loss_func_mask[3] and len(supervised_indices) > 0):
					loss_4 = log_likelihood_loss_supervised(self.theta, self.pi, sample[1][supervised_indices], sample[2][supervised_indices], sample[3][supervised_indices], self.k, self.n_classes, self.continuous_mask, self.qc)
				else:
					loss_4 = 0

				if(self.loss_func_mask[4]):
					loss_5 = log_likelihood_loss(self.theta, self.pi, sample[2][unsupervised_indices], sample[3][unsupervised_indices], self.k, self.n_classes, self.continuous_mask, self.qc)
				else:
					loss_5 =0

				if(self.loss_func_mask[5]):
					if(len(supervised_indices) >0):
						supervised_indices = supervised_indices.tolist()
						probs_graphical = probability(self.theta, self.pi, torch.cat([sample[2][unsupervised_indices], sample[2][supervised_indices]]),\
						torch.cat([sample[3][unsupervised_indices],sample[3][supervised_indices]]), self.k, self.n_classes, self.continuous_mask, self.qc)
					else:
						probs_graphical = probability(self.theta, self.pi,sample[2][unsupervised_indices],sample[3][unsupervised_indices],\
							self.k, self.n_classes, self.continuous_mask, self.qc)
					probs_graphical = (probs_graphical.t() / probs_graphical.sum(1)).t()
					probs_fm = torch.nn.softmax(dim = 1)(self.feature_model(sample[0]))
					loss_6 = kl_divergence(probs_fm, probs_graphical)
				else:
					loss_6= 0

				if(self.loss_func_mask[6]):
					prec_loss = precision_loss(self.theta, self.k, self.n_classes, self.qt)
				else:
					prec_loss =0

				loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6 + prec_loss
				if loss != 0:
					loss.backward()
					optimizer_gm.step()
					optimizer_fm.step()

			#gm test
			y_pred = predict_gm_labels(self.theta, self.pi, self.l_test, self.s_test, self.k, self.n_classes, self.continuous_mask, self.qc)
			if self.use_accuracy_score:
				gm_test_acc = accuracy_score(self.y_test, y_pred)
			else:
				gm_test_acc = f1_score(self.y_test, y_pred, average = self.metric_avg)
			gm_test_prec = prec_score(self.y_test, y_pred, average = self.metric_avg)
			gm_test_recall = recall_score(self.y_test, y_pred, average = self.metric_avg)

			#gm validation
			y_pred = predict_gm_labels(self.theta, self.pi, self.l_valid, self.s_valid, self.k, self.n_classes, self.continuous_mask, self.qc)
			if self.use_accuracy_score:
				gm_valid_acc = accuracy_score(self.y_valid, y_pred)
			else:
				gm_valid_acc = f1_score(self.y_valid, y_pred, average = self.metric_avg)

			#fm test
			probs = torch.nn.softmax(dim = 1)(self.feature_model(self.x_test))
			y_pred = np.argmax(probs.detach().numpy(), 1)
			if self.use_accuracy_score:
				fm_test_acc = accuracy_score(self.y_test, y_pred)
			else:
				fm_test_acc = f1_score(self.y_test, y_pred, average = self.metric_avg)
			fm_test_prec = prec_score(self.y_test, y_pred, average = self.metric_avg)
			fm_test_recall = recall_score(self.y_test, y_pred, average = self.metric_avg)

			#fm validation
			probs = torch.nn.softmax(dim = 1)(self.feature_model(self.x_valid))
			y_pred = np.argmax(probs.detach().numpy(), 1)
			if self.use_accuracy_score:
				fm_valid_acc = accuracy_score(self.y_valid, y_pred)
			else:
				fm_valid_acc = f1_score(self.y_valid, y_pred, average = self.metric_avg)

			if path_log != None:
				file.write("{}: Epoch: {}\tgm_valid_score: {}\tfm_valid_score: {}\n".format(self.score_used, epoch, gm_valid_acc, fm_valid_acc))
				if epoch % 5 == 0:
					file.write("{}: Epoch: {}\tgm_test_score: {}\tfm_test_score: {}\n".format(self.score_used, epoch, gm_test_acc, fm_test_acc))

			if epoch > self.start_len and gm_valid_acc >= best_score_gm_val and gm_valid_acc >= best_score_fm_val:
				if gm_valid_acc == best_score_gm_val or gm_valid_acc == best_score_fm_val:
					if best_score_gm_test < gm_test_acc or best_score_fm_test < fm_test_acc:
						best_epoch = epoch
						self.pi_optimal = (self.pi).detach().clone()
						self.theta_optimal = (self.theta).detach().clone()
						self.fm_optimal_params = deepcopy((self.feature_model).state_dict())

						best_score_fm_val = fm_valid_acc
						best_score_fm_test = fm_test_acc
						best_score_gm_val = gm_valid_acc
						best_score_gm_test = gm_test_acc

						best_prec_fm_test = fm_test_prec
						best_recall_fm_test  = fm_test_recall
						best_prec_gm_test = gm_test_prec
						best_recall_gm_test  = gm_test_recall
				else:
					best_epoch = epoch
					self.pi_optimal = (self.pi).detach().clone()
					self.theta_optimal = (self.theta).detach().clone()
					self.fm_optimal_params = deepcopy((self.feature_model).state_dict())

					best_score_fm_val = fm_valid_acc
					best_score_fm_test = fm_test_acc
					best_score_gm_val = gm_valid_acc
					best_score_gm_test = gm_test_acc

					best_prec_fm_test = fm_test_prec
					best_recall_fm_test  = fm_test_recall
					best_prec_gm_test = gm_test_prec
					best_recall_gm_test  = gm_test_recall
					stop_early_fm = []
					stop_early_gm = []

			if epoch > self.start_len and fm_valid_acc >= best_score_fm_val and fm_valid_acc >= best_score_gm_val:
				if fm_valid_acc == best_score_fm_val or fm_valid_acc == best_score_gm_val:
					if best_score_fm_test < fm_test_acc or best_score_gm_test < gm_test_acc:
						best_epoch = epoch
						self.pi_optimal = (self.pi).detach().clone()
						self.theta_optimal = (self.theta).detach().clone()
						self.fm_optimal_params = deepcopy((self.feature_model).state_dict())

						best_score_fm_val = fm_valid_acc
						best_score_fm_test = fm_test_acc
						best_score_gm_val = gm_valid_acc
						best_score_gm_test = gm_test_acc

						best_prec_fm_test = fm_test_prec
						best_recall_fm_test  = fm_test_recall
						best_prec_gm_test = gm_test_prec
						best_recall_gm_test  = gm_test_recall
				else:
					best_epoch = epoch
					self.pi_optimal = (self.pi).detach().clone()
					self.theta_optimal = (self.theta).detach().clone()
					self.fm_optimal_params = deepcopy((self.feature_model).state_dict())
					
					best_score_fm_val = fm_valid_acc
					best_score_fm_test = fm_test_acc
					best_score_gm_val = gm_valid_acc
					best_score_gm_test = gm_test_acc

					best_prec_fm_test = fm_test_prec
					best_recall_fm_test  = fm_test_recall
					best_prec_gm_test = gm_test_prec
					best_recall_gm_test  = gm_test_recall
					stop_early_fm = []
					stop_early_gm = []

			if len(stop_early_fm) > self.stop_len and len(stop_early_gm) > self.stop_len and (all(best_score_fm_val >= k for k in stop_early_fm) or \
			all(best_score_gm_val >= k for k in stop_early_gm)):
				self.stopped_early = True
				break
			else:
				stop_early_fm.append(fm_valid_acc)
				stop_early_gm.append(gm_valid_acc)

		#epoch for loop ended

		self.is_training_done = True

		if self.stopped_early:
			print('early stopping... best_epoch: {}'.format(best_epoch))
		else:
			print('best_epoch: {}'.format(best_epoch))
		
		print('best_gm_val_score:{}\tbest_fm_val_score:{}\n'.format(\
			best_score_gm_val, best_score_fm_val))
		print('best_gm_test_score:{}\tbest_fm_test_score:{}\n'.format(\
			best_score_gm_test, best_score_fm_test))
		print('best_gm_test_precision:{}\tbest_fm_test_precision:{}\n'.format(\
			best_prec_gm_test, best_prec_fm_test))
		print('best_gm_test_recall:{}\tbest_fm_test_recall:{}\n'.format(\
			best_recall_gm_test, best_recall_fm_test))
			

		# Algo ended

		# below prints and writes to file, the final test accuracies
		# print("final_gm_test_acc: {}\tfinal_fm_test_acc: {}\n".format(gm_test_acc, fm_test_acc))
		# if path_log != None:
		# 	file.write("final_test_acc: {}\tfinal_fm_test_acc: {}\n".format(gm_test_acc, fm_test_acc))
		# 	file.close()

		(self.feature_model).load_state_dict(self.fm_optimal_params)
		(self.feature_model).eval()

		if return_gm:
			return (torch.nn.softmax(dim = 1)(self.feature_model(self.x_unsup))), \
				probability(self.theta_optimal, self.pi_optimal, self.l_unsup, self.s_unsup, self.k, self.n_classes, self.continuous_mask, self.qc)
		else:
			return (torch.nn.softmax(dim = 1)(self.feature_model(self.x_unsup)))

	def fit_and_predict(self, loss_func_mask, batch_size, lr_fm, lr_gm, path_log = None, return_gm = False, n_epochs = 100, start_len = 5,\
	 stop_len = 10, is_qt = True, is_qc = True, qt = 0.9, qc = 0.85, n_hidden = 512, feature_model = 'nn', metric_avg = 'macro', need_strings = False):
		'''
		Args:
			loss_func_mask: list/numpy array of size 7 or (7,) where loss_func_mask[i] should be 1 if Loss function (i+1) should be included, 0 else. Checkout Eq(3) in :cite:p:`2020:JL`
			batch_size: Batch size, type should be integer
			lr_fm: Learning rate for feature model, type is integer or float
			lr_gm: Learning rate for graphical model(cage), type is integer or float
			path_log: Path to log file
			return_gm: Return the predictions of graphical model? the allowed values are True, False. Default value is False
			n_epochs: Number of epochs in each run, type is integer, default is 100
			start_len: A parameter used in validation, type is integer, default is 5
			stop_len: A parameter used in validation, type is integer, default is 10
			is_qt: True if quality guide is available(and will be provided in 'qt' argument). False if quality guide is intended to be found from validation instances. Default is True
			is_qc: True if quality index is available(and will be provided in 'qc' argument). False if quality index is intended to be found from validation instances. Default is True
			qt: Quality guide of shape (n_lfs,) of type numpy.ndarray OR a float. Values must be between 0 and 1. Default is 0.9
			qc: Quality index of shape (n_lfs,) of type numpy.ndarray OR a float. Values must be between 0 and 1. Default is 0.85
			n_hidden: Number of hidden layer nodes if feature model is 'nn', type is integer, default is 512
			feature_model: The model intended to be used for features, allowed values are 'lr'(Logistic Regression) or 'nn'(Neural network with 1 hidden layer) string, default is 'nn'
			metric_avg: Average metric to be used in calculating f1_score/precision/recall, default is 'macro'
			need_strings: If True, the output will be in the form of strings(class names). Else it is in the form of class values(given to classes in Enum). Default is False

		Return:
			If return_gm is True; the return value is two predicted labels of numpy array of shape (num_instances, ), first one is through feature model, other one through graphical model.
			Else; the return value is predicted labels of numpy array of shape (num_instances,) through feature model. 
		'''
		assert type(need_strings) == np.bool
		if return_gm:
			proba_1, proba_2 = self.fit_and_predict_proba(loss_func_mask, batch_size, lr_fm, lr_gm, path_log, return_gm, n_epochs, start_len,\
	 		stop_len, is_qt, is_qc, qt, qc, n_hidden, feature_model, metric_avg)
			return get_predictions(proba_1, self.class_map, self.class_dict, need_strings), get_predictions(proba_2, self.class_map, self.class_dict, need_strings)
		else:
			proba = self.fit_and_predict_proba(loss_func_mask, batch_size, lr_fm, lr_gm, path_log, return_gm, n_epochs, start_len,\
	 		stop_len, is_qt, is_qc, qt, qc, n_hidden, feature_model, metric_avg)
			return get_predictions(proba, self.class_map, self.class_dict, need_strings)

	
	def predict_gm_proba(self, path_test):
		'''
			Used to find the predicted labels based on the trained parameters of graphical model(CAGE)

		Args:
			path_test: Path to the pickle file containing test data set
		
		Return:
			numpy.ndarray of shape (num_instances, num_classes) where i,j-th element is the probability of ith instance being the jth class(the jth value when sorted in ascending order of values in Enum)
			[Note: no aggregration/algorithm-running will be done using the current input]
		'''
		assert self.is_training_done

		data = get_data(path_test, True, self.class_map)
		s_test = torch.tensor(data[6]).double()
		s_test[s_test > 0.999] = 0.999
		s_test[s_test < 0.001] = 0.001
		assert (data[2]).shape[1] == self.n_lfs
		m_test = torch.abs(torch.tensor(data[2]).long())

		return probability(self.theta_optimal, self.pi_optimal, m_test, s_test, self.k, self.n_classes, self.continuous_mask, self.qc)

	def predict_fm_proba(self, path_test):
		'''
			Used to find the predicted labels based on the trained parameters of feature model

		Args:
			path_test: Path to the pickle file containing test data set
		
		Return:
			numpy.ndarray of shape (num_instances, num_classes) where i,j-th element is the probability of ith instance being the jth class(the jth value when sorted in ascending order of values in Enum)
			[Note: no aggregration/algorithm-running will be done using the current input]
		'''
		assert self.is_training_done

		data = get_data(path_test, True, self.class_map)
		x_test = data[0]

		return (torch.nn.softmax(dim = 1)(self.feature_model(x_test)))

	def predict_gm(self, path_test, need_strings = False):
		'''
			Used to find the predicted labels based on the trained parameters of graphical model(CAGE)

		Args:
			path_test: Path to the pickle file containing test data set
			need_strings: If True, the output will be in the form of strings(class names). Else it is in the form of class values(given to classes in Enum). Default is False
		
		Return:
			numpy.ndarray of shape (num_instances,) which are predicted labels. Elements are numbers/strings depending on need_strings attribute is false/true resp.
			[Note: no aggregration/algorithm-running will be done using the current input]
		'''
		assert type(need_strings) == np.bool
		return get_predictions(self.predict_gm_proba(path_test), self.class_map, self.class_dict, need_strings)

	def predict_fm(self, path_test, need_strings = False):
		'''
			Used to find the predicted labels based on the trained parameters of feature model

		Args:
			path_test: Path to the pickle file containing test data set
			need_strings: If True, the output will be in the form of strings(class names). Else it is in the form of class values(given to classes in Enum). Default is False

		Return:
			numpy.ndarray of shape (num_instances,) which are predicted labels. Elements are numbers/strings depending on need_strings attribute is false/true resp.
			[Note: no aggregration/algorithm-running will be done using the current input]
		'''
		assert type(need_strings) == np.bool
		return get_predictions(self.predict_fm_proba(path_test), self.class_map, self.class_dict, need_strings)


