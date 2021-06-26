import torch
from torch import optim
import pickle
from os import path as check_path
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from ..utils.data_editor import get_data, get_classes, get_predictions
from ..utils.utils_cage import probability, log_likelihood_loss, precision_loss, predict_gm_labels

class Cage:
	'''
	Cage class:
		Class for Data Programming using CAGE
		[Note: from here on, graphical model(gm) and CAGE algorithm terms are used interchangeably]

	Args:
		path_json: Path to json file consisting of number to string(class name) map
		n_lfs: number of labelling functions used to generate pickle files

	'''
	def __init__(self, path_json, n_lfs):
		assert type(path_json) == str
		assert type(n_lfs) == np.int or type(n_lfs) == np.float

		use_cuda = torch.cuda.is_available()
		self.device = torch.device("cuda" if use_cuda else "cpu")
		torch.backends.cudnn.benchmark = True

		self.class_dict = get_classes(path_json)
		self.class_list = list((self.class_dict).keys())
		self.class_list.sort()
		self.n_classes = len(self.class_dict)

		self.class_map = {value: index for index, value in enumerate(self.class_list)}
		self.class_map[None] = self.n_classes

		self.n_lfs = int(n_lfs)
		self.n, self.k = None, None #continuous_mask, labels of LFs

		self.pi = torch.ones((self.n_classes, self.n_lfs), device = self.device).double()
		(self.pi).requires_grad = True

		self.theta = torch.ones((self.n_classes, self.n_lfs), device = self.device).double()
		(self.theta).requires_grad = True

	def save_params(self, save_path):
		'''
			member function to save parameters of Cage

		Args:
			save_path: path to pickle file to save parameters
		'''
		file_ = open(save_path, 'wb')
		pickle.dump(self.theta, file_)
		pickle.dump(self.pi, file_)
		pickle.dump(self.n_classes, file_)
		pickle.dump(self.n_lfs, file_)
		file_.close()
		return

	def load_params(self, load_path):
		'''
			member function to load parameters to Cage

		Args:
			load_path: path to pickle file to load parameters
		'''
		assert check_path.exists(load_path)
		file_ = open(load_path, 'rb')
		self.theta = pickle.load(file_)
		self.pi = pickle.load(file_)
		assert self.n_classes == pickle.load(file_)
		assert self.n_lfs == pickle.load(file_)
		file_.close()

		assert (self.pi).shape == (self.n_classes, self.n_lfs)
		assert (self.theta).shape == (self.n_classes, self.n_lfs)
		return

	def fit_and_predict_proba(self, path_pkl, path_test = None, path_log = None, qt = 0.9, qc = 0.85, metric_avg = ['binary'], n_epochs = 100, lr = 0.01):
		'''
		Args:
			path_pkl: Path to pickle file of input data in standard format
			path_test: Path to the pickle file containing test data in standard format
			path_log: Path to log file. No log is produced if path_test is None. Default is None which prints accuracies/f1_scores is printed to terminal
			qt: Quality guide of shape (n_lfs,) of type numpy.ndarray OR a float. Values must be between 0 and 1. Default is 0.9
			qc: Quality index of shape (n_lfs,) of type numpy.ndarray OR a float. Values must be between 0 and 1. Default is 0.85
			metric_avg: List of average metric to be used in calculating f1_score, default is ['binary']. Use None for not calculating f1_score
			n_epochs:Number of epochs, default is 100
			lr: Learning rate for torch.optim, default is 0.01

		Return:
			numpy.ndarray of shape (num_instances, num_classes) where i,j-th element is the probability of ith instance being the jth class(the jth value when sorted in ascending order of values in Enum)
		'''
		assert  type(path_pkl) == str
		assert (type(qt) == np.float and (qt >= 0 and qt <= 1)) or (type(qt) == np.ndarray and (np.all(np.logical_and(qt>=0, qt<=1)) ) )\
		 or (type(qt) == np.int and (qt == 0 or qt == 1))

		assert (type(qc) == np.float and (qc >= 0 and qc <= 1)) or (type(qc) == np.ndarray and (np.all(np.logical_and(qc>=0, qc<=1)) ) )\
		 or (type(qc) == np.int and (qc == 0 or qc == 1))

		for temp in metric_avg:
			assert temp in ['micro', 'macro', 'samples','weighted', 'binary'] or metric_avg is None
		assert type(n_epochs) == np.int or type(n_epochs) == np.float
		assert type(lr) == np.int or type(lr) == np.float

		data = get_data(path_pkl, True, self.class_map)
		m = torch.abs(torch.tensor(data[2], device = self.device).long())
		s = torch.tensor(data[6], device = self.device).double() # continuous score
		if self.n == None:	
			self.n = torch.tensor(data[7], device = self.device).double() # Mask for s/continuous_mask
		else:
			assert torch.all(torch.tensor(data[7], device = self.device).double().eq(self.n))
		if self.k == None:
			self.k = torch.tensor(data[8], device = self.device).long() # LF's classes
		else:
			assert torch.all(torch.tensor(data[8], device = self.device).long().eq(self.k))
		s[s > 0.999] = 0.999 # clip s
		s[s < 0.001] = 0.001 # clip s

		assert self.n_lfs == m.shape[1]
		assert self.n_classes == data[9]

		qt_ = torch.tensor(qt, device = self.device).double() if type(qt) == np.ndarray else (torch.ones(self.n_lfs, device = self.device).double() * qt)
		qc_ = torch.tensor(qc, device = self.device).double() if type(qc) == np.ndarray else qc
		metric_avg_ = list(set(metric_avg))
		n_epochs_ = int(n_epochs)
		
		optimizer = optim.Adam([self.theta, self.pi], lr=lr, weight_decay=0)

		file = None
		if path_test != None and path_log != None:
			file = open(path_log, "a+")
			file.write("CAGE log:\tn_classes: {}\tn_LFs: {}\tn_epochs: {}\tlr: {}\n".format(self.n_classes, self.n_lfs, n_epochs, lr))
		elif path_test != None:
			print("CAGE log:\tn_classes: {}\tn_LFs: {}\tn_epochs: {}\tlr: {}".format(self.n_classes, self.n_lfs, n_epochs, lr))

		y_true_test = None
		s_test, m_test = None, None
		if path_test != None:
			data = get_data(path_test, True, self.class_map)
			m_test, y_true_test, s_test = data[2], data[3], data[6]
			assert m_test.shape[0] == y_true_test.shape[0]
			y_true_test = y_true_test.flatten()
			assert self.n_lfs == m_test.shape[1]
			assert self.n_classes == data[9]
			assert torch.all(torch.tensor(data[7], device = self.device).double().eq(self.n))
			assert torch.all(torch.tensor(data[8], device = self.device).long().eq(self.k))

		assert np.all(np.logical_and(y_true_test >= 0, y_true_test < self.n_classes))

		with tqdm(total=n_epochs_) as pbar:
			for epoch in range(n_epochs_):
				optimizer.zero_grad()
				loss = log_likelihood_loss(self.theta, self.pi, m, s, self.k, self.n_classes, self.n, qc_, self.device)
				prec_loss = precision_loss(self.theta, self.k, self.n_classes, qt_, self.device)
				loss += prec_loss

				if path_test != None:
					y_pred = self.__predict_specific(m_test, s_test, qc_)
					if path_log != None:
						file.write("Epoch: {}\ttest_accuracy_score: {}\n".format(epoch, accuracy_score(y_true_test, y_pred)))
					else:
						print("Epoch: {}\ttest_accuracy_score: {}".format(epoch, accuracy_score(y_true_test, y_pred)))
					if epoch == n_epochs_-1:
						print("final_test_accuracy_score: {}".format(accuracy_score(y_true_test, y_pred)))
					for temp in metric_avg_:
						if path_log != None:
							file.write("Epoch: {}\ttest_average_metric: {}\ttest_f1_score: {}\n".format(epoch, temp, f1_score(y_true_test, y_pred, average = temp)))
						else:
							print("Epoch: {}\ttest_average_metric: {}\ttest_f1_score: {}".format(epoch, temp, f1_score(y_true_test, y_pred, average = temp)))
						if epoch == n_epochs_-1:
							print("test_average_metric: {}\tfinal_test_f1_score: {}".format(temp, f1_score(y_true_test, y_pred, average = temp)))

				loss.backward()
				optimizer.step()
				pbar.update()

		if path_test != None and path_log != None:
			file.close()

		return (probability(self.theta, self.pi, m, s, self.k, self.n_classes, self.n, qc_, self.device)).cpu().detach().numpy()

	def fit_and_predict(self, path_pkl, path_test = None, path_log = None, qt = 0.9, qc = 0.85, metric_avg = ['binary'], n_epochs = 100, lr = 0.01, need_strings = False):
		'''
		Args:
			path_pkl: Path to pickle file of input data in standard format
			path_test: Path to the pickle file containing test data in standard format
			path_log: Path to log file. No log is produced if path_test is None. Default is None which prints accuracies/f1_scores is printed to terminal
			qt: Quality guide of shape (n_lfs,) of type numpy.ndarray OR a float. Values must be between 0 and 1. Default is 0.9
			qc: Quality index of shape (n_lfs,) of type numpy.ndarray OR a float. Values must be between 0 and 1. Default is 0.85
			metric_avg: List of average metric to be used in calculating f1_score, default is ['binary']
			n_epochs:Number of epochs, default is 100
			lr: Learning rate for torch.optim, default is 0.01
			need_strings: If True, the output will be in the form of strings(class names). Else it is in the form of class values(given to classes in Enum). Default is False

		Return:
			numpy.ndarray of shape (num_instances,) which are aggregated/predicted labels. Elements are numbers/strings depending on need_strings attribute is false/true resp.
		'''
		assert type(need_strings) == np.bool
		proba = self.fit_and_predict_proba(path_pkl, path_test, path_log, qt, qc, metric_avg, n_epochs, lr)
		return get_predictions(proba, self.class_map, self.class_dict, need_strings)

	def __predict_specific(self, m_test, s_test, qc_):
		'''
			Used to predict labels based on s_test and m_test

		Args:
			m_test: numpy arrays of shape (num_instances, num_rules), m_test[i][j] is 1 if jth LF is triggered on ith instance, else it is 0
			s_test: numpy arrays of shape (num_instances, num_rules), s_test[i][j] is the continuous score of jth LF on ith instance
			qc_: Quality index of shape (n_lfs,) of type numpy.ndarray OR a float. Values must be between 0 and 1
		
		Return:
			numpy.ndarray of shape (num_instances,) which are predicted labels. Note that here the class labels appearing may not be the ones used in the Enum
			[Note: no aggregration/algorithm-running will be done using the current input]
		'''
		s_temp = torch.tensor(s_test, device = self.device).double()
		s_temp[s_temp > 0.999] = 0.999
		s_temp[s_temp < 0.001] = 0.001
		assert m_test.shape == s_test.shape
		assert m_test.shape[1] == self.n_lfs
		assert np.all(np.logical_or(m_test == 1, m_test == 0))
		m_temp = torch.abs(torch.tensor(m_test, device = self.device).long())
		return predict_gm_labels(self.theta, self.pi, m_temp, s_temp, self.k, self.n_classes, self.n, qc_, self.device)

	def predict_proba(self, path_test, qc = 0.85):
		'''
			Used to predict labels based on a pickle file with path path_test

		Args:
			path_test: Path to the pickle file containing test data set in standard format
			qc: Quality index of shape (n_lfs,) of type numpy.ndarray OR a float. Values must be between 0 and 1. Default is 0.85

		Return:
			numpy.ndarray of shape (num_instances, num_classes) where i,j-th element is the probability of ith instance being the jth class(the jth value when sorted in ascending order of values in Enum)
			[Note: no aggregration/algorithm-running will be done using the current input]
		'''
		assert (type(qc) == np.float and (qc >= 0 and qc <= 1)) or (type(qc) == np.ndarray and (np.all(np.logical_and(qc>=0, qc<=1)) ) )\
		 or (type(qc) == np.int and (qc == 0 or qc == 1))
		data = get_data(path_test, True, self.class_map)
		assert (data[2]).shape[1] == self.n_lfs and data[9] == self.n_classes
		temp_k = torch.tensor(data[8], device = self.device).long()
		assert self.k == None or torch.all(temp_k.eq(self.k))
		temp_n = torch.tensor(data[7], device = self.device).double()
		assert self.n == None or torch.all(temp_n.eq(self.n))
		s_test = torch.tensor(data[6], device = self.device).double()
		s_test[s_test > 0.999] = 0.999
		s_test[s_test < 0.001] = 0.001
		m_test = torch.abs(torch.tensor(data[2], device = self.device).long())	

		qc_ = torch.tensor(qc).double() if type(qc) == np.ndarray else qc
		if self.n == None or self.k == None:
			print("Warning: Predict is used before training any paramters in Cage class. Hope you have loaded parameters.")
		return (probability(self.theta, self.pi, m_test, s_test, temp_k, self.n_classes, temp_n, qc_, self.device)).cpu().detach().numpy()
		
	def predict(self, path_test, qc = 0.85, need_strings = False):
		'''
			Used to predict labels based on a pickle file with path path_test
			
		Args:
			path_test: Path to the pickle file containing test data set in standard format
			qc: Quality index of shape (n_lfs,) of type numpy.ndarray OR a float. Values must be between 0 and 1. Default is 0.85
			need_strings: If True, the output will be in the form of strings(class names). Else it is in the form of class values(given to classes in Enum). Default is False

		Return:
			numpy.ndarray of shape (num_instances,) which are predicted labels. Elements are numbers/strings depending on need_strings attribute is false/true resp.
			[Note: no aggregration/algorithm-running will be done using the current input]
		'''
		assert type(need_strings) == np.bool
		return get_predictions(self.predict_proba(path_test, qc), self.class_map, self.class_dict, need_strings)