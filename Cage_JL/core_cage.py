import torch
from torch import optim
import numpy as np
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from utils import *

class Cage:
	'''
	Cage class:
		Class for Data Programming using CAGE
		[Note: from here on, graphical model(gm) imply CAGE algorithm]

	Args:
		n_classes: Number of classes/labels, type is integer
		path: Path to pickle file of input data
		metric_avg: List of average metric to be used in calculating f1_score, default is ['binary']
		n_epochs:Number of epochs, default is 100
		lr: Learning rate for torch.optim, default is 0.01
		qt: Quality guide of shape (n_lfs,) of type numpy.ndarray OR a float. Values must be between 0 and 1
		qc: Quality index of shape (n_lfs,) of type numpy.ndarray OR a float. Values must be between 0 and 1

	'''
	def __init__(self, n_classes, path, metric_avg = ['binary'], n_epochs = 100, lr = 0.01, qt = None, qc = None):
		assert type(n_classes) == np.int or type(n_classes) == np.float
		assert type(path) == str
		assert os.path.exists(path)
		for temp in metric_avg:
			assert temp in ['micro', 'macro', 'samples','weighted', 'binary'] or metric_avg is None
		assert type(n_epochs) == np.int or type(n_epochs) == np.float
		assert type(lr) == np.int or type(lr) == np.float

		if type(qt) == float:
			assert qt >= 0 and qt <= 1
		elif type(qt) == np.ndarray:
			assert np.all(np.logical_and(qt>=0, qt<=1))
		elif type(qt) == np.int:
			assert qt == 0 or qt == 1
		else:
			print("core_cage: Invalid type for qt in Cage class")
			exit(1)

		if type(qc) == float:
			assert qc >= 0 and qc <= 1
		elif type(qc) == np.ndarray:
			assert np.all(np.logical_and(qc>=0, qc<=1))
		elif type(qc) == np.int:
			assert qc == 0 or qc == 1
		else:
			print("core_cage: Invalid type for qc in Cage class")
			exit(1)

		data = get_data(path)

		self.l = torch.abs(torch.tensor(data[2]).long())
		self.s = torch.tensor(data[6]).double() # continuous score
		self.n = torch.tensor(data[7]).double() # Mask for s/continuous_mask
		self.k = torch.tensor(data[8]).long() # LF's classes
		self.s[self.s > 0.999] = 0.999 # clip s
		self.s[self.s < 0.001] = 0.001 # clip s

		self.n_classes = int(n_classes)
		self.metric_avg = list(set(metric_avg))
		self.n_epochs = int(n_epochs)
		self.lr = lr
		self.n_lfs = self.l.shape[1]

		self.n_instances, self.n_features = data[0].shape

		self.qt = torch.tensor(qt).double() if qt != None and type(qt) == np.ndarray else \
		((torch.ones(self.n_lfs).double() * qt) if qt != None else (torch.ones(self.n_lfs).double() * 0.9)) 
		self.qc = torch.tensor(qt).double() if qc != None and type(qc) == np.ndarray else (qc if qc != None else 0.85)

		self.pi = torch.ones((n_classes, n_lfs)).double()
		(self.pi).requires_grad = True

		self.theta = torch.ones((n_classes, n_lfs)).double()
		(self.theta).requires_grad = True

	def fit(self, path_test = None, path_log = None):
		'''
		Args:
			path_test: Path to the pickle file containing test data set
			path_log: Path to log file, default value is None. No log is producede if path is None
		Return:
			numpy.ndarray of shape (num_instances,) which are aggregated/predicted labels
		'''
		use_cuda = torch.cuda.is_available()
		device = torch.device("cuda:0" if use_cuda else "cpu")
		torch.backends.cudnn.benchmark = True

		optimizer = optim.Adam([self.theta, self.pi], lr=self.lr, weight_decay=0)

		file = None
		if path_test != None and path_log != None:
			file = open(path_log, "a+")

		y_true_test = None
		s_test, m_test = None, None
		if path_test != None:
			data = get_data(path_test)
			m_test, y_true_test, s_test = data[2], data[3], data[6]

		assert np.all(np.logical_and(y_true_test >= 0, y_true_test < self.n_classes))

		for epoch in range(self.n_epochs):
			optimizer.zero_grad()
			loss = log_likelihood_loss(self.theta, self.pi, self.l, self.s, self.k, self.n_classes, self.n, self.qc)
			prec_loss = precision_loss(self.theta, self.k, self.n_classes, self.qt)
			loss += prec_loss

			if path_test != None and path_log != None:
				y_pred = predict_specific(s_test, m_test)
				file.write("Epoch: {}\taccuracy_score: {}".format(epoch, accuracy_score(y_true_test, y_pred)))
				for temp in self.metric_avg:
					file.write("Epoch: {}\tmetric_avg: {}\tf1_score: {}".format(epoch, temp, f1_score(y_true_test, y_pred, average = temp)))

			loss.backward()
			optimizer.step()

		if path_test != None and path_log != None:
			file.close()

		return predict_gm(self.theta, self.pi, self.l, self.s, self.k, self.n_classes, self.n, self.qc)

	def predict_specific(self, s_test, m_test):
		'''
		Args:
			s_test: numpy arrays of shape (num_instances, num_rules), s_test[i][j] is the continuous score of jth LF on ith instance
			m_test: numpy arrays of shape (num_instances, num_rules), m_test[i][j] is 1 if jth LF is triggered on ith instance, else it is 0
		Return:
			numpy.ndarray of shape (num_instances,) which are predicted labels
			[Note: no aggregration/algorithm-running will be done using the current input]
		'''
		s_temp = torch.tensor(s_test).double()
		s_temp[s_temp > 0.999] = 0.999
		s_temp[s_temp < 0.001] = 0.001
		assert m_test.shape == s_test.shape
		assert m_test.shape[1] == self.n_lfs
		assert np.all(np.logical_or(m_test == 1 or m_test == 0))
		m_temp = torch.abs(torch.tensor(m_test).long())
		return predict_gm(self.theta, self.pi, m_temp, s_temp, self.k, self.n_classes, self.n, self.qc)

	def predict(self, path_test):
		'''
		Args:
			path_test: Path to the pickle file containing test data set
		Return:
			numpy.ndarray of shape (num_instances,) which are predicted labels
			[Note: no aggregration/algorithm-running will be done using the current input]
		'''
		data = get_data(path_test)
		s_test = torch.tensor(data[6]).double()
		s_test[s_test > 0.999] = 0.999
		s_test[s_test < 0.001] = 0.001
		assert (data[2]).shape[1] == self.n_lfs
		m_test = torch.abs(torch.tensor(data[2]).long())

		return predict_gm(self.theta, self.pi, m_test, s_test, self.k, self.n_classes, self.n, self.qc)