import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
import pickle
from os import path as check_path
from copy import deepcopy
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score as prec_score
from sklearn.metrics import recall_score as recall_score

from ..utils.data_editor import get_data, get_classes, get_predictions
from ..utils.utils_cage import probability, log_likelihood_loss, precision_loss, predict_gm_labels
from ..utils.utils_jl import log_likelihood_loss_supervised, entropy, kl_divergence
from .models.models import *

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

class JL:
	'''
	Joint_Learning class:
		[Note: from here on, feature model(fm) and feature-based classification model are used interchangeably. graphical model(gm) and CAGE algorithm terms are used interchangeably]

		Loss function number | Calculated over | Loss function: (useful for loss_func_mask in fit_and_predict_proba and fit_and_predict functions)

			1, L, Cross Entropy(prob_from_feature_model, true_labels)

			2, U, Entropy(prob_from_feature_model)

			3, U, Cross Entropy(prob_from_feature_model, prob_from_graphical_model)

			4, L, Negative Log Likelihood

			5, U, Negative Log Likelihood(marginalised over true labels)

			6, L and U, KL Divergence(prob_feature_model, prob_graphical_model)

			7, _,  Quality guide
	
	Args:
		path_json: Path to json file containing the dictionary of number to string(class name) map
		n_lfs: number of labelling functions used to generate pickle files
		n_features: number of features for each instance in the first array of pickle file aka feature matrix
		feature_model: The model intended to be used for features, allowed values are 'lr'(Logistic Regression) or 'nn'(Neural network with 2 hidden layer) string, default is 'nn'
		n_hidden: Number of hidden layer nodes if feature model is 'nn', type is integer, default is 512
	'''
	def __init__(self, path_json, n_lfs, n_features, feature_model = 'nn', n_hidden = 512):
		assert type(path_json) == str
		assert type(n_lfs) == np.int or type(n_lfs) == np.float
		assert type(n_features) == np.int or type(n_features) == np.float
		assert type(n_hidden) == np.int or type(n_hidden) == np.float
		assert feature_model == 'lr' or feature_model == 'nn'
		
		use_cuda = torch.cuda.is_available()
		self.device = torch.device("cuda" if use_cuda else "cpu")
		torch.backends.cudnn.benchmark = True
		torch.set_default_dtype(torch.float64)

		self.class_dict = get_classes(path_json)
		self.class_list = list((self.class_dict).keys())
		self.class_list.sort()
		self.n_classes = len(self.class_dict)

		self.class_map = {value: index for index, value in enumerate(self.class_list)}
		self.class_map[None] = self.n_classes

		self.n_lfs = int(n_lfs)
		self.n_hidden = int(n_hidden)
		self.feature_based_model = feature_model
		self.n_features = n_features
		self.k, self.continuous_mask = None, None

		self.pi = torch.ones((self.n_classes, self.n_lfs), device = self.device).double()
		(self.pi).requires_grad = True
		self.theta = torch.ones((self.n_classes, self.n_lfs), device = self.device).double()
		(self.theta).requires_grad = True

		if self.feature_based_model == 'lr':
			self.feature_model = LogisticRegression(self.n_features, self.n_classes).to(device = self.device)
		elif self.feature_based_model =='nn':
			self.feature_model = DeepNet(self.n_features, self.n_hidden, self.n_classes).to(device = self.device)
		else:
			print('Error: JL class - unrecognised feature_model in initialisation')
			exit(1)

		self.fm_optimal_params = deepcopy((self.feature_model).state_dict())
		self.pi_optimal, self.theta_optimal = (self.pi).detach().clone(), (self.theta).detach().clone()

	def save_params(self, save_path):
		'''
			member function to save parameters of JL

		Args:
			save_path: path to pickle file to save parameters
		'''
		file_ = open(save_path, 'wb')
		pickle.dump(self.theta, file_)
		pickle.dump(self.pi, file_)
		pickle.dump((self.feature_model).state_dict(), file_)
		pickle.dump(self.theta_optimal, file_)
		pickle.dump(self.pi_optimal, file_)
		pickle.dump((self.fm_optimal_params), file_)
		pickle.dump(self.n_classes, file_)
		pickle.dump(self.n_lfs, file_)
		pickle.dump(self.n_features, file_)
		pickle.dump(self.n_hidden, file_)
		pickle.dump(self.feature_based_model, file_)
		file_.close()
		return

	def load_params(self, load_path):
		'''
			member function to load parameters to JL

		Args:
			load_path: path to pickle file to load parameters
		'''
		assert check_path.exists(load_path)
		file_ = open(load_path, 'rb')
		self.theta = pickle.load(file_)
		self.pi = pickle.load(file_)
		fm_params = pickle.load(file_)
		(self.feature_model).load_state_dict(fm_params)

		self.theta_optimal = pickle.load(file_)
		self.pi_optimal = pickle.load(file_)
		self.fm_optimal_params = pickle.load(file_)

		assert self.n_classes == pickle.load(file_)
		assert self.n_lfs == pickle.load(file_)
		assert self.n_features == pickle.load(file_)
		temp_n_hidden = pickle.load(file_)
		temp_feature_based_model = pickle.load(file_)
		assert self.feature_based_model == temp_feature_based_model
		if temp_feature_based_model == 'nn':
			assert self.n_hidden == temp_n_hidden
		
		file_.close()

		assert (self.pi).shape == (self.n_classes, self.n_lfs)
		assert (self.theta).shape == (self.n_classes, self.n_lfs)
		assert (self.pi_optimal).shape == (self.n_classes, self.n_lfs)
		assert (self.theta_optimal).shape == (self.n_classes, self.n_lfs)

		return

	def fit_and_predict_proba(self, path_L, path_U, path_V, path_T, loss_func_mask, batch_size, lr_fm, lr_gm, use_accuracy_score, path_log = None, return_gm = False, n_epochs = 100, start_len = 7,\
	 stop_len = 10, is_qt = True, is_qc = True, qt = 0.9, qc = 0.85, metric_avg = 'binary'):
		'''
		Args:
			path_L: Path to pickle file of labelled instances
			path_U: Path to pickle file of unlabelled instances
			path_V: Path to pickle file of validation instances
			path_T: Path to pickle file of test instances
			loss_func_mask: list of size 7 where loss_func_mask[i] should be 1 if Loss function (i+1) should be included, 0 else. Checkout Eq(3) in :cite:p:`DBLP:journals/corr/abs-2008-09887`
			batch_size: Batch size, type should be integer
			lr_fm: Learning rate for feature model, type is integer or float
			lr_gm: Learning rate for graphical model(cage algorithm), type is integer or float
			use_accuracy_score: The score to use for termination condition on validation set. True for accuracy_score, False for f1_score
			path_log: Path to log file to append log. Default is None which prints accuracies/f1_scores is printed to terminal
			return_gm: Return the predictions of graphical model? the allowed values are True, False. Default value is False
			n_epochs: Number of epochs in each run, type is integer, default is 100
			start_len: A parameter used in validation, refers to the least epoch after which validation checks need to be performed, type is integer, default is 7
			stop_len: A parameter used in validation, refers to the least number of continuous epochs of non incresing validation accuracy after which the training should be stopped, type is integer, default is 10
			is_qt: True if quality guide is available(and will be provided in 'qt' argument). False if quality guide is intended to be found from validation instances. Default is True
			is_qc: True if quality index is available(and will be provided in 'qc' argument). False if quality index is intended to be found from validation instances. Default is True
			qt: Quality guide of shape (n_lfs,) of type numpy.ndarray OR a float. Values must be between 0 and 1. Default is 0.9
			qc: Quality index of shape (n_lfs,) of type numpy.ndarray OR a float. Values must be between 0 and 1. Default is 0.85
			metric_avg: Average metric to be used in calculating f1_score/precision/recall, default is 'binary'

		Return:
			If return_gm is True; the return value is two predicted labels of numpy array of shape (num_instances, num_classes), first one is through feature model, other one through graphical model.
			Else; the return value is predicted labels of numpy array of shape (num_instances, num_classes) through feature model. For a given model i,j-th element is the probability of ith instance being the 
			jth class(the jth value when sorted in ascending order of values in Enum) using that model. It is suggested to use the probailities of feature model
		'''
		assert type(path_L) == str and type(path_V) == str and type(path_V) == str and type(path_T) == str
		assert type(return_gm) == np.bool
		assert (type(loss_func_mask) == list) and len(loss_func_mask) == 7
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
		assert metric_avg in ['micro', 'macro', 'samples', 'weighted', 'binary']

		batch_size_ = int(batch_size)
		n_epochs_ = int(n_epochs)
		start_len_ = int(start_len)
		stop_len_ = int(stop_len)

		score_used = "accuracy_score" if use_accuracy_score else "f1_score"

		assert start_len_ <= n_epochs_ and stop_len <= n_epochs_

		data_L = get_data(path_L, True, self.class_map)
		data_U = get_data(path_U, True, self.class_map)
		data_V = get_data(path_V, True, self.class_map)
		data_T = get_data(path_T, True, self.class_map)

		assert data_L[9] == self.n_classes and data_U[9] == data_L[9] and data_V[9] == data_L[9] and data_T[9] == data_L[9]

		x_sup = torch.tensor(data_L[0]).double()
		y_sup = torch.tensor(data_L[3]).long()
		l_sup = torch.tensor(data_L[2]).long()
		s_sup = torch.tensor(data_L[6]).double()

		excluding = []
		temp_index = 0
		for temp in data_U[1]:
			if(np.all(temp == int(self.n_classes)) ):
				excluding.append(temp_index)
			temp_index+=1

		x_unsup = torch.tensor(np.delete(data_U[0], excluding, axis=0)).double()
		y_unsup = torch.zeros((x_unsup).shape[0]).long()
		l_unsup = torch.tensor(np.delete(data_U[2], excluding, axis=0)).long()
		s_unsup = torch.tensor(np.delete(data_U[6], excluding, axis=0)).double()

		x_valid = torch.tensor(data_V[0]).double()
		y_valid = data_V[3]
		l_valid = torch.tensor(data_V[2]).long()
		s_valid = torch.tensor(data_V[6]).double()

		x_test = torch.tensor(data_T[0]).double()
		y_test = data_T[3]
		l_test = torch.tensor(data_T[2]).long()
		s_test = torch.tensor(data_T[6]).double()

		y_sup = (y_sup).view(-1)
		y_valid = (y_valid).flatten()
		y_test = (y_test).flatten()

		assert self.n_features == x_sup.shape[1]
		assert self.n_lfs == l_sup.shape[1]
		if self. k == None:
			self.k = torch.tensor(data_L[8], device = self.device).long() # LF's classes
		else:
			assert torch.all(torch.tensor(data_L[8], device = self.device).double().eq(self.k))
		if self.continuous_mask == None:
			self.continuous_mask = torch.tensor(data_L[7], device = self.device).double() # Mask for s/continuous_mask
		else:
			assert torch.all(torch.tensor(data_L[7], device = self.device).double().eq(self.continuous_mask))

		assert np.all(data_L[8] == data_U[8]) and np.all(data_L[8] == data_V[8]) and np.all(data_L[8] == data_T[8])
		assert np.all(data_L[7] == data_U[7]) and np.all(data_L[7] == data_V[7]) and np.all(data_L[7] == data_T[7])

		assert x_sup.shape[1] == self.n_features and x_unsup.shape[1] == self.n_features \
		 and x_valid.shape[1] == self.n_features and x_test.shape[1] == self.n_features
		assert x_sup.shape[0] == y_sup.shape[0] and x_sup.shape[0] == l_sup.shape[0]\
		 and l_sup.shape == s_sup.shape and l_sup.shape[1] == self.n_lfs
		assert x_unsup.shape[0] == y_unsup.shape[0] and x_unsup.shape[0] == l_unsup.shape[0]\
		 and l_unsup.shape == s_unsup.shape and l_unsup.shape[1] == self.n_lfs
		assert x_valid.shape[0] == y_valid.shape[0] and x_valid.shape[0] == l_valid.shape[0]\
		 and l_valid.shape == s_valid.shape and l_valid.shape[1] == self.n_lfs
		assert x_test.shape[0] == y_test.shape[0] and x_test.shape[0] == l_test.shape[0]\
		 and l_test.shape == s_test.shape and l_test.shape[1] == self.n_lfs

		# clipping s
		s_sup[s_sup > 0.999] = 0.999
		s_sup[s_sup < 0.001] = 0.001
		s_unsup[s_unsup > 0.999] = 0.999
		s_unsup[s_unsup < 0.001] = 0.001
		s_valid[s_valid > 0.999] = 0.999
		s_valid[s_valid < 0.001] = 0.001
		s_test[s_test > 0.999] = 0.999
		s_test[s_test < 0.001] = 0.001

		l = torch.cat([l_sup, l_unsup])
		s = torch.cat([s_sup, s_unsup])
		x_train = torch.cat([x_sup, x_unsup])
		y_train = torch.cat([y_sup, y_unsup])
		supervised_mask = torch.cat([torch.ones(l_sup.shape[0]), torch.zeros(l_unsup.shape[0])])

		if is_qt:
			qt_ = torch.tensor(qt, device = self.device).double() if type(qt) == np.ndarray else (torch.ones(self.n_lfs, device = self.device).double() * qt)
		else:
			prec_lfs=[]
			for i in range(self.n_lfs):
				correct = 0
				for j in range(len(y_valid)):
					if y_valid[j] == l_valid[j][i]:
						correct+=1
				prec_lfs.append(correct/len(y_valid))
			qt_ = torch.tensor(prec_lfs).double()

		if is_qc:
			qc_ = torch.tensor(qc, device = self.device).double() if type(qc) == np.ndarray else qc
		else:
			qc_ = torch.tensor(np.mean(s_valid, axis = 0), device = self.device)

		file = None
		if path_log != None:
			file = open(path_log, "a+")
			file.write("JL log:\tn_classes: {}\tn_LFs: {}\tn_features: {}\tn_hidden: {}\tfeature_model:{}\tlr_fm: {}\tlr_gm:{}\tuse_accuracy_score: {}\tn_epochs:{}\tstart_len: {}\tstop_len:{}\n".format(\
				self.n_classes, self.n_lfs, self.n_features, self.n_hidden, self.feature_based_model, lr_fm, lr_gm, use_accuracy_score, n_epochs, start_len, stop_len))
		else:
			print("JL log:\tn_classes: {}\tn_LFs: {}\tn_features: {}\tn_hidden: {}\tfeature_model:{}\tlr_fm: {}\tlr_gm:{}\tuse_accuracy_score: {}\tn_epochs:{}\tstart_len: {}\tstop_len:{}".format(\
				self.n_classes, self.n_lfs, self.n_features, self.n_hidden, self.feature_based_model, lr_fm, lr_gm, use_accuracy_score, n_epochs, start_len, stop_len))

		#Algo starting
		optimizer_fm = torch.optim.Adam(self.feature_model.parameters(), lr = lr_fm)
		optimizer_gm = torch.optim.Adam([self.theta, self.pi], lr = lr_gm, weight_decay=0)
		supervised_criterion = torch.nn.CrossEntropyLoss()

		dataset = TensorDataset(x_train, y_train, l, s, supervised_mask)
		loader = DataLoader(dataset, batch_size = batch_size_, shuffle = True, drop_last = False, pin_memory = True)

		best_score_fm_test, best_score_gm_test, best_epoch, best_score_fm_val, best_score_gm_val = 0,0,0,0,0
		best_prec_fm_test, best_recall_fm_test, best_prec_gm_test, best_recall_gm_test= 0,0,0,0

		gm_test_acc, fm_test_acc = -1, -1

		stopped_epoch = -1
		stop_early_fm, stop_early_gm = [], []

		with tqdm(total=n_epochs_) as pbar:
			for epoch in range(n_epochs_):
				
				self.feature_model.train()

				for _, sample in enumerate(loader):
					optimizer_fm.zero_grad()
					optimizer_gm.zero_grad()

					for i in range(len(sample)):
						sample[i] = sample[i].to(device = self.device)

					supervised_indices = sample[4].nonzero().view(-1)
					unsupervised_indices = (1-sample[4]).nonzero().squeeze()

					if(loss_func_mask[0]):
						if len(supervised_indices) > 0:
							loss_1 = supervised_criterion(self.feature_model(sample[0][supervised_indices]), sample[1][supervised_indices])
						else:
							loss_1 = 0
					else:
						loss_1 = 0

					if(loss_func_mask[1]):
						unsupervised_fm_probability = torch.nn.Softmax(dim = 1)(self.feature_model(sample[0][unsupervised_indices]))
						loss_2 = entropy(unsupervised_fm_probability)
					else:
						loss_2 = 0

					if(loss_func_mask[2]):
						y_pred_unsupervised = predict_gm_labels(self.theta, self.pi, sample[2][unsupervised_indices], sample[3][unsupervised_indices], self.k, self.n_classes, self.continuous_mask, qc_, self.device)
						loss_3 = supervised_criterion(self.feature_model(sample[0][unsupervised_indices]), torch.tensor(y_pred_unsupervised, device = self.device))
					else:
						loss_3 = 0

					if (loss_func_mask[3] and len(supervised_indices) > 0):
						loss_4 = log_likelihood_loss_supervised(self.theta, self.pi, sample[1][supervised_indices], sample[2][supervised_indices], sample[3][supervised_indices], self.k, self.n_classes, self.continuous_mask, qc_, self.device)
					else:
						loss_4 = 0

					if(loss_func_mask[4]):
						loss_5 = log_likelihood_loss(self.theta, self.pi, sample[2][unsupervised_indices], sample[3][unsupervised_indices], self.k, self.n_classes, self.continuous_mask, qc_, self.device)
					else:
						loss_5 = 0

					if(loss_func_mask[5]):
						if(len(supervised_indices) >0):
							supervised_indices = supervised_indices.tolist()
							probs_graphical = probability(self.theta, self.pi, torch.cat([sample[2][unsupervised_indices], sample[2][supervised_indices]]),\
							torch.cat([sample[3][unsupervised_indices],sample[3][supervised_indices]]), self.k, self.n_classes, self.continuous_mask, qc_, self.device)
						else:
							probs_graphical = probability(self.theta, self.pi,sample[2][unsupervised_indices],sample[3][unsupervised_indices],\
								self.k, self.n_classes, self.continuous_mask, qc_, self.device)
						probs_graphical = (probs_graphical.t() / probs_graphical.sum(1)).t()
						probs_fm = torch.nn.Softmax(dim = 1)(self.feature_model(sample[0]))
						loss_6 = kl_divergence(probs_fm, probs_graphical)
					else:
						loss_6 = 0

					if(loss_func_mask[6]):
						prec_loss = precision_loss(self.theta, self.k, self.n_classes, qt_, self.device)
					else:
						prec_loss = 0

					loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6 + prec_loss
					if loss != 0:
						loss.backward()
						optimizer_gm.step()
						optimizer_fm.step()

				#gm test
				y_pred = predict_gm_labels(self.theta, self.pi, l_test.to(device = self.device), s_test.to(device = self.device), self.k, self.n_classes, self.continuous_mask, qc_, self.device)
				if use_accuracy_score:
					gm_test_acc = accuracy_score(y_test, y_pred)
				else:
					gm_test_acc = f1_score(y_test, y_pred, average = metric_avg)
				gm_test_prec = prec_score(y_test, y_pred, average = metric_avg)
				gm_test_recall = recall_score(y_test, y_pred, average = metric_avg)

				#gm validation
				y_pred = predict_gm_labels(self.theta, self.pi, l_valid.to(device = self.device), s_valid.to(device = self.device), self.k, self.n_classes, self.continuous_mask, qc_, self.device)
				if use_accuracy_score:
					gm_valid_acc = accuracy_score(y_valid, y_pred)
				else:
					gm_valid_acc = f1_score(y_valid, y_pred, average = metric_avg)

				(self.feature_model).eval()

				#fm test
				probs = torch.nn.Softmax(dim = 1)(self.feature_model(x_test.to(device = self.device)))
				y_pred = np.argmax(probs.cpu().detach().numpy(), 1)
				if use_accuracy_score:
					fm_test_acc = accuracy_score(y_test, y_pred)
				else:
					fm_test_acc = f1_score(y_test, y_pred, average = metric_avg)
				fm_test_prec = prec_score(y_test, y_pred, average = metric_avg)
				fm_test_recall = recall_score(y_test, y_pred, average = metric_avg)

				#fm validation
				probs = torch.nn.Softmax(dim = 1)(self.feature_model(x_valid.to(device = self.device)))
				y_pred = np.argmax(probs.cpu().detach().numpy(), 1)
				if use_accuracy_score:
					fm_valid_acc = accuracy_score(y_valid, y_pred)
				else:
					fm_valid_acc = f1_score(y_valid, y_pred, average = metric_avg)

				(self.feature_model).train()

				if path_log != None:
					file.write("{}: Epoch: {}\tgm_valid_score: {}\tfm_valid_score: {}\n".format(score_used, epoch, gm_valid_acc, fm_valid_acc))
					if epoch % 5 == 0:
						file.write("{}: Epoch: {}\tgm_test_score: {}\tfm_test_score: {}\n".format(score_used, epoch, gm_test_acc, fm_test_acc))
				else:
					print("{}: Epoch: {}\tgm_valid_score: {}\tfm_valid_score: {}".format(score_used, epoch, gm_valid_acc, fm_valid_acc))
					if epoch % 5 == 0:
						print("{}: Epoch: {}\tgm_test_score: {}\tfm_test_score: {}".format(score_used, epoch, gm_test_acc, fm_test_acc))

				if epoch > start_len_ and gm_valid_acc >= best_score_gm_val and gm_valid_acc >= best_score_fm_val:
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

				if epoch > start_len_ and fm_valid_acc >= best_score_fm_val and fm_valid_acc >= best_score_gm_val:
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

				if len(stop_early_fm) > stop_len_ and len(stop_early_gm) > stop_len_ and (all(best_score_fm_val >= k for k in stop_early_fm) or \
				all(best_score_gm_val >= k for k in stop_early_gm)):
					stopped_epoch = epoch
					break
				else:
					stop_early_fm.append(fm_valid_acc)
					stop_early_gm.append(gm_valid_acc)

				pbar.update()
				#epoch for loop ended


		if stopped_epoch == -1:
			print('best_epoch: {}'.format(best_epoch))
		else:
			print('early stopping at epoch: {}\tbest_epoch: {}'.format(stopped_epoch, best_epoch))

		if use_accuracy_score:
			print('score used: accuracy_score')
		else:
			print('score used: f1_score')
		
		print('best_gm_val_score:{}\tbest_fm_val_score:{}'.format(\
			best_score_gm_val, best_score_fm_val))
		print('best_gm_test_score:{}\tbest_fm_test_score:{}'.format(\
			best_score_gm_test, best_score_fm_test))
		print('best_gm_test_precision:{}\tbest_fm_test_precision:{}'.format(\
			best_prec_gm_test, best_prec_fm_test))
		print('best_gm_test_recall:{}\tbest_fm_test_recall:{}'.format(\
			best_recall_gm_test, best_recall_fm_test))
			

		# Algo ended

		# below prints and writes to file, the final test accuracies
		# print("final_gm_test_acc: {}\tfinal_fm_test_acc: {}\n".format(gm_test_acc, fm_test_acc))
		if path_log != None:
		# 	file.write("final_test_acc: {}\tfinal_fm_test_acc: {}\n".format(gm_test_acc, fm_test_acc))
			file.close()

		(self.feature_model).load_state_dict(self.fm_optimal_params)
		(self.feature_model).eval()
		fm_predictions = (torch.nn.Softmax(dim = 1)(self.feature_model(torch.tensor(data_U[0], device = self.device).double()) )).cpu().detach().numpy()
		(self.feature_model).train()

		if return_gm:
			return fm_predictions, (probability(self.theta_optimal, self.pi_optimal, torch.tensor(data_U[2], device = self.device).long(), torch.tensor(data_U[6], device = self.device).double(), \
				self.k, self.n_classes, self.continuous_mask, qc_, self.device)).cpu().detach().numpy()
		else:
			return fm_predictions

	def fit_and_predict(self, path_L, path_U, path_V, path_T, loss_func_mask, batch_size, lr_fm, lr_gm, use_accuracy_score, path_log = None, return_gm = False, n_epochs = 100, start_len = 7,\
	 stop_len = 10, is_qt = True, is_qc = True, qt = 0.9, qc = 0.85, metric_avg = 'binary', need_strings = False):
		'''
		Args:
			path_L: Path to pickle file of labelled instances
			path_U: Path to pickle file of unlabelled instances
			path_V: Path to pickle file of validation instances
			path_T: Path to pickle file of test instances
			loss_func_mask: list of size 7 where loss_func_mask[i] should be 1 if Loss function (i+1) should be included, 0 else. Checkout Eq(3) in :cite:p:`DBLP:journals/corr/abs-2008-09887`
			batch_size: Batch size, type should be integer
			lr_fm: Learning rate for feature model, type is integer or float
			lr_gm: Learning rate for graphical model(cage algorithm), type is integer or float
			use_accuracy_score: The score to use for termination condition on validation set. True for accuracy_score, False for f1_score
			path_log: Path to log file to append log. Default is None which prints accuracies/f1_scores is printed to terminal
			return_gm: Return the predictions of graphical model? the allowed values are True, False. Default value is False
			n_epochs: Number of epochs in each run, type is integer, default is 100
			start_len: A parameter used in validation, refers to the least epoch after which validation checks need to be performed, type is integer, default is 7
			stop_len: A parameter used in validation, refers to the least number of continuous epochs of non incresing validation accuracy after which the training should be stopped, type is integer, default is 10
			is_qt: True if quality guide is available(and will be provided in 'qt' argument). False if quality guide is intended to be found from validation instances. Default is True
			is_qc: True if quality index is available(and will be provided in 'qc' argument). False if quality index is intended to be found from validation instances. Default is True
			qt: Quality guide of shape (n_lfs,) of type numpy.ndarray OR a float. Values must be between 0 and 1. Default is 0.9
			qc: Quality index of shape (n_lfs,) of type numpy.ndarray OR a float. Values must be between 0 and 1. Default is 0.85
			metric_avg: Average metric to be used in calculating f1_score/precision/recall, default is 'binary'
			need_strings: If True, the output will be in the form of strings(class names). Else it is in the form of class values(given to classes in Enum). Default is False

		Return:
			If return_gm is True; the return value is two predicted labels of numpy array of shape (num_instances, ), first one is through feature model, other one through graphical model.
			Else; the return value is predicted labels of numpy array of shape (num_instances,) through feature model. It is suggested to use the probailities of feature model
		'''
		assert type(need_strings) == np.bool
		if return_gm:
			proba_1, proba_2 = self.fit_and_predict_proba(path_L, path_U, path_V, path_T, loss_func_mask, batch_size, lr_fm, lr_gm, use_accuracy_score, path_log, return_gm, n_epochs, start_len,\
	 		stop_len, is_qt, is_qc, qt, qc, metric_avg)
			return get_predictions(proba_1, self.class_map, self.class_dict, need_strings), get_predictions(proba_2, self.class_map, self.class_dict, need_strings)
		else:
			proba = self.fit_and_predict_proba(path_L, path_U, path_V, path_T, loss_func_mask, batch_size, lr_fm, lr_gm, use_accuracy_score, path_log, return_gm, n_epochs, start_len,\
	 		stop_len, is_qt, is_qc, qt, qc, metric_avg)
			return get_predictions(proba, self.class_map, self.class_dict, need_strings)

	
	def predict_gm_proba(self, path_test, qc = 0.85):
		'''
			Used to find the predicted labels based on the trained parameters of graphical model(CAGE)

		Args:
			path_test: Path to the pickle file containing test data set
			qc: Quality index of shape (n_lfs,) of type numpy.ndarray OR a float. Values must be between 0 and 1. Default is 0.85
		
		Return:
			numpy.ndarray of shape (num_instances, num_classes) where i,j-th element is the probability of ith instance being the jth class(the jth value when sorted in ascending order of values in Enum)
			[Note: no aggregration/algorithm-running will be done using the current input]. It is suggested to use the probailities of feature model
		'''
		assert (type(qc) == np.float and (qc >= 0 and qc <= 1)) or (type(qc) == np.ndarray and (np.all(np.logical_and(qc>=0, qc<=1)) ) )\
		 or (type(qc) == np.int and (qc == 0 or qc == 1))

		data = get_data(path_test, True, self.class_map)
		s_test = torch.tensor(data[6], device = self.device).double()
		s_test[s_test > 0.999] = 0.999
		s_test[s_test < 0.001] = 0.001
		assert (data[2]).shape[1] == self.n_lfs and data[9] == self.n_classes
		assert (data[0].shape)[1] == self.n_features
		temp_k = torch.tensor(data[8], device = self.device).long()
		assert self.k == None or torch.all(temp_k.eq(self.k))
		temp_continuous_mask = torch.tensor(data[7], device = self.device).double()
		assert self.continuous_mask == None or torch.all(temp_continuous_mask.eq(self.continuous_mask))
		m_test = torch.abs(torch.tensor(data[2], device = self.device).long())
		qc_ = torch.tensor(qc, device = self.device).double() if type(qc) == np.ndarray else qc

		if self.continuous_mask == None or self.k == None:
			print("Warning: Predict is used before training any paramters in JL class. Hope you have loaded parameters.")
		return (probability(self.theta_optimal, self.pi_optimal, m_test, s_test, temp_k, self.n_classes, temp_continuous_mask, qc_, self.device)).cpu().detach().numpy()
		
	def predict_fm_proba(self, x_test):
		'''
			Used to find the predicted labels based on the trained parameters of feature model

		Args:
			x_test: numpy array of shape (num_instances, num_features) containing data whose labels are to be predicted
		
		Return:
			numpy.ndarray of shape (num_instances, num_classes) where i,j-th element is the probability of ith instance being the jth class(the jth value when sorted in ascending order of values in Enum)
			[Note: no aggregration/algorithm-running will be done using the current input]. It is suggested to use the probailities of feature model
		'''
		assert x_test.shape[1] == self.n_features

		if self.continuous_mask == None or self.k == None:
			print("Warning: Predict is used before training any paramters in JL class. Hope you have loaded parameters.")

		(self.feature_model).load_state_dict(self.fm_optimal_params)
		(self.feature_model).eval()
		fm_predictions = (torch.nn.Softmax(dim = 1)(self.feature_model(torch.tensor(x_test, device = self.device).double()))).cpu().detach().numpy()
		(self.feature_model).train()

		return fm_predictions

	def predict_gm(self, path_test, qc = 0.85, need_strings = False):
		'''
			Used to find the predicted labels based on the trained parameters of graphical model(CAGE)

		Args:
			path_test: Path to the pickle file containing test data set
			qc: Quality index of shape (n_lfs,) of type numpy.ndarray OR a float. Values must be between 0 and 1. Default is 0.85
			need_strings: If True, the output will be in the form of strings(class names). Else it is in the form of class values(given to classes in Enum). Default is False
		
		Return:
			numpy.ndarray of shape (num_instances,) which are predicted labels. Elements are numbers/strings depending on need_strings attribute is false/true resp.
			[Note: no aggregration/algorithm-running will be done using the current input]. It is suggested to use the probailities of feature model
		'''
		assert type(need_strings) == np.bool
		return get_predictions(self.predict_gm_proba(path_test, qc), self.class_map, self.class_dict, need_strings)

	def predict_fm(self, x_test, need_strings = False):
		'''
			Used to find the predicted labels based on the trained parameters of feature model

		Args:
			x_test: numpy array of shape (num_instances, num_features) containing data whose labels are to be predicted
			need_strings: If True, the output will be in the form of strings(class names). Else it is in the form of class values(given to classes in Enum). Default is False

		Return:
			numpy.ndarray of shape (num_instances,) which are predicted labels. Elements are numbers/strings depending on need_strings attribute is false/true resp.
			[Note: no aggregration/algorithm-running will be done using the current input]. It is suggested to use the probailities of feature model
		'''
		assert type(need_strings) == np.bool
		return get_predictions(self.predict_fm_proba(x_test), self.class_map, self.class_dict, need_strings)


