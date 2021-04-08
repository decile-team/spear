import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sklearn

from utils import *
from utils_jl import *
from models import *


class Joint_Learning:
	'''
	Joint_Learning class:
		[Note: from here on, feature model is short for feature-based classification model and graphical model(gm) imply CAGE algorithm]
	
	Args:
		n_classes: Number of classes/labels, type is integer
		path_L: Path to pickle file of labelled instances
		path_U: Path to pickle file of unlabelled instances
		path_V: Path to pickle file of validation instances
		path_T: Path to pickle file of test instances
		loss_func_mask: list/numpy array of size 7 or (7,) where loss_func_mask[i] should be 1 if Loss function (i+1) should be included, 0 else
		is_qt: True if quality guide is available. False if quality guide is intended to be found from validation instances
		is_qc: True if quality index is available. False if quality index is intended to be found from validation instances
		batch_size: Batch size, type should be integer
		lr_feature: Learning rate for feature model, type is integer or float
		lr_gm: Learning rate for graphical model(cage), type is integer or float
		use_accuracy_score: The score to use for termination condition on validation set. True for accuracy_score, False for f1_score
		feature_model: The model intended to be used for features, allowed values are 'lr' or 'nn' string, default is 'nn'
		metric_avg: Average metric to be used in calculating f1_score/precision/recall, default is 'macro'
		qt: Quality guide of shape (n_lfs,) of type numpy.ndarray OR a float. Values must be between 0 and 1
		qc: Quality index of shape (n_lfs,) of type numpy.ndarray OR a float. Values must be between 0 and 1
		n_hidden: Number of hidden layer nodes if feature model is 'nn', type is integer, default is 512
		n_epochs: Number of epochs in each run, type is integer, default is 100
		n_runs: Number of runs ,type is integer, default is 10
		start_len: A parameter used in validation, type is integer, default is 5
		stop_len: A parameter used in validation, type is integer, default is 10
	'''
	def __init__(self, n_classes, path_L, path_U, path_V, path_T , loss_func_mask, is_qt, is_qc, batch_size, lr_feature, lr_gm,\
	 use_accuracy_score, feature_model = 'nn', metric_avg = 'macro', qt = None, qc = None, n_hidden = 512, n_epochs = 100, n_runs = 10, start_len = 5, stop_len = 10):

		assert type(n_classes) == np.int or type(n_classes) == np.float
		assert type(path_L) == str and type(path_V) == str and type(path_V) == str and type(path_T) == str and type(metric_avg) == str
		assert os.path.exists(path_L) and os.path.exists(path_U) and os.path.exists(path_V) and os.path.exists(path_T)
		assert len(loss_func_mask) == 7
		assert feature_model == 'lr' or feature_model == 'nn'
		assert type(is_qt) == np.bool and type(is_qc) == np.bool and type(use_accuracy_score) == np.bool
		assert type(batch_size) == np.int or type(batch_size) == np.float
		assert type(lr_feature) == np.int or type(lr_feature) == np.float
		assert type(lr_gm) == np.int or type(lr_gm) == np.float

		if n_lfs != None:
			assert type(n_lfs) == np.int
		if type(qt) == float:
			assert qt >= 0 and qt <= 1
		elif type(qt) == np.ndarray:
			assert np.all(np.logical_and(qt>=0, qt<=1))
		elif type(qt) == np.int:
			assert qt == 0 or qt == 1
		else:
			print("Invalid type for qt in Joint_Learning class")
			exit(1)

		if n_lfs != None:
			assert type(n_lfs) == np.int
		if type(qc) == float:
			assert qc >= 0 and qc <= 1
		elif type(qc) == np.ndarray:
			assert np.all(np.logical_and(qc>=0, qc<=1))
		elif type(qc) == np.int:
			assert qc == 0 or qc == 1
		else:
			print("Invalid type for qc in Joint_Learning class")
			exit(1)

		assert type(n_hidden) == np.int or type(n_hidden) == np.float
		assert type(n_epochs) == np.int or type(n_epochs) == np.float
		assert type(n_runs) == np.int or type(n_runs) == np.float
		assert type(start_len) == np.int or type(start_len) == np.float
		assert type(stop_len) == np.int or type(stop_len) == np.float

		torch.set_default_dtype(torch.float64)
		self.n_classes = int(n_classes)
		self.loss_func_mask = loss_func_mask
		self.feature_model = feature_model
		self.batch_size = int(batch_size)
		self.n_hidden = int(n_hidden)
		self.lr_feature = lr_feature
		self.lr_gm = lr_gm
		self.use_accuracy_score = use_accuracy_score
		self.metric_avg = metric_avg
		self.n_epochs = int(n_epochs)
		self.n_runs = int(n_runs)
		self.start_len = int(start_len)
		self.stop_len = int(stop_len)

		assert self.start_len <= self.n_epochs and self.stop_len <= self.n_epochs

		if self.use_accuracy_score:
			from sklearn.metrics import accuracy_score as score
		else:
			from sklearn.metrics import f1_score as score
		from sklearn.metrics import precision_score as prec_score
		from sklearn.metrics import recall_score as recall_score

		data_L = get_data(path_L)
		data_U = get_data(path_U)
		data_V = get_data(path_V)
		data_T = get_data(path_T)

		self.x_sup = torch.tensor(data_L[0]).double()
		self.y_sup = torch.tensor(data_L[3]).long()
		self.l_sup = torch.tensor(data_L[2]).long()
		self.s_sup = torch.tensor(data_L[6]).double()

		excluding = []
		temp_index = 0
		for temp in data_U[1]:
			if(np.all(temp == int(self.n_classes)) ):
				excluding.append(index_temp)
			index_temp+=1

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

		self.n_features = self.x_sup.shape[1]
		self.k = torch.tensor(data_L[8]).long() # LF's classes
		self.n_lfs = self.l_sup.shape[1]
		self.continuous_mask = torch.tensor(data_L[7]).double() # Mask for s/continuous_mask

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

		assert self.x_sup.shape[1] == self.n_features and self.x_unsup.shape[1] == self.n_features \
		 and self.x_valid.shape[1] == self.n_features and self.x_test.shape[1] == self.n_features

		assert self.x_sup.shape[0] == self.y_sup.shape[0] and self.x_sup.shape[0] == self.l_sup.shape[0]\
		 and self.l_sup.shape == self.s_sup.shape and self.l_sup.shape[1] == self.n_lfs
		assert self.x_unsup.shape[0] == self.y_unsup.shape[0] and self.x_unsup.shape[0] == self.l_unsup.shape[0]\
		 and self.l_unsup.shape == self.s_unsup.shape and self.l_unsup.shape[1] == self.n_lfs
		assert self.x_valid.shape[0] == self.y_valid.shape[0] and self.x_valid.shape[0] == self.l_valid.shape[0]\
		 and self.l_valid.shape == self.s_valid.shape and self.l_valid.shape[1] == self.n_lfs
		assert self.x_test.shape[0] == self.y_test.shape[0] and self.x_test.shape[0] == self.l_test.shape[0]\
		 and self.l_test.shape == self.s_test.shape and self.l_test.shape[1] == self.n_lfs

		if is_qt:
			self.qt = torch.tensor(qt).double() if qt != None and type(qt) == np.ndarray else \
			((torch.ones(self.n_lfs).double() * qt) if qt != None else (torch.ones(self.n_lfs).double() * 0.9)) 
		else:
			prec_lfs=[]
			for i in range(self.n_lfs):
				correct = 0
				for j in range(len(y_valid)):
					if y_valid[j] == l_valid[j][i]:
						correct+=1
				prec_lfs.append(correct/len(y_valid))
			self.qt = torch.tensor(prec_lfs).double()

		if is_qg:
			self.qc = torch.tensor(qt).double() if qc != None and type(qc) == np.ndarray else (qc if qc != None else 0.85)
		else:
			self.qc = torch.tensor(np.mean(s_valid, axis = 0))

		# clip s
		self.s_sup[self.s_sup > 0.999] = 0.999
		self.s_sup[self.s_sup < 0.001] = 0.001
		self.s_unsup[self.s_unsup > 0.999] = 0.999
		self.s_unsup[self.s_unsup < 0.001] = 0.001
		self.s_valid[self.s_valid > 0.999] = 0.999
		self.s_valid[self.s_valid < 0.001] = 0.001
		self.s_test[self.s_test > 0.999] = 0.999
		self.s_test[self.s_test < 0.001] = 0.001

		self.l = torch.cat([l_sup, l_unsup])
		self.s = torch.cat([s_sup, s_unsup])
		self.x_train = torch.cat([x_sup, x_unsup])
		self.y_train = torch.cat([y_sup, y_unsup])
		self.supervised_mask = torch.cat([torch.ones(l_sup.shape[0]), torch.zeros(l_unsup.shape[0])])

		self.pi = torch.ones((self.n_classes, self.n_lfs)).double()
		self.theta = torch.ones((self.n_classes, self.n_lfs)).double()

		if self.feature_model == 'lr':
			self.lr_model = LogisticRegression(self.n_features, self.n_classes)
		elif self.feature_model =='nn':
			self.lr_model = DeepNet(n_features, self.n_hidden, self.n_classes)

	def fit(self, return_gm = False, path = None):
		'''
		Args:
			return_gm: Return the predictions of graphical model? the allowed values are True, False. Default value is False
			path: Path to log file
		Return: 
			If return_gm is True; the return value is two predicted labels of numpy array of shape (num_instances,), first one is through graphical model, other one through feature model.
			Else; the return value is predicted labels of numpy array of shape (num_instances,) through feature model
		'''

		assert type(return_gm) == np.bool

		file = None
		if path != None:
			file = open(path, "a+")

		use_cuda = torch.cuda.is_available()
		device = torch.device("cuda:0" if use_cuda else "cpu")
		torch.backends.cudnn.benchmark = True

		final_score_gm, final_score_lr, final_score_gm_val, final_score_lr_val = [],[],[],[]
		final_score_lr_prec, final_score_lr_recall, final_score_gm_prec, final_score_gm_recall = [],[],[],[]
		for run in range(0,self.n_runs):
			self.pi = torch.ones((self.n_classes, self.n_lfs)).double()
			(self.pi).requires_grad = True

			self.theta = torch.ones((self.n_classes, self.n_lfs)).double() * 1
			(self.theta).requires_grad = True

			optimizer = torch.optim.Adam([{"params": self.lr_model.parameters()}, {"params": [self.pi, self.theta]}], lr=0.001)
			optimizer_lr = torch.optim.Adam(self.lr_model.parameters(), lr = self.lr_feature)
			optimizer_gm = torch.optim.Adam([self.theta, self.pi], lr = self.lr_gm, weight_decay=0)
			supervised_criterion = torch.nn.CrossEntropyLoss()

			dataset = TensorDataset(self.x_train, self.y_train, self.l, self.s, self.supervised_mask)

			loader = DataLoader(dataset, batch_size = self.batch_size, shuffle = True, pin_memory = True)

			best_score_lr,best_score_gm,best_epoch_lr,best_epoch_gm,best_score_lr_val, best_score_gm_val = 0,0,0,0,0,0
			best_score_lr_prec,best_score_lr_recall ,best_score_gm_prec,best_score_gm_recall= 0,0,0,0

			stop_early_lr, stop_early_gm = [], []

			for epoch in range(self.n_epochs):
				self.lr_model.train()

				for batch_ndx, sample in enumerate(loader):
					optimizer_lr.zero_grad()
					optimizer_gm.zero_grad()

					unsup = []
					sup = []
					supervised_indices = sample[4].nonzero().view(-1)
					# unsupervised_indices = indices  # Uncomment for entropy
					unsupervised_indices = (1-sample[4]).nonzero().squeeze()

					if(self.loss_func_mask[0]):
						if len(supervised_indices) > 0:
							loss_1 = supervised_criterion(self.lr_model(sample[0][supervised_indices]), sample[1][supervised_indices])
						else:
							loss_1 = 0
					else:
						loss_1=0

					if(self.loss_func_mask[1]):
						unsupervised_lr_probability = torch.nn.Softmax()(self.lr_model(sample[0][unsupervised_indices]))
						loss_2 = entropy(unsupervised_lr_probability)
					else:
						loss_2=0

					if(self.loss_func_mask[2]):
						y_pred_unsupervised = predict_gm(self.theta, self.pi, sample[2][unsupervised_indices], sample[3][unsupervised_indices], self.k, self.n_classes, self.continuous_mask, self.qc)
						loss_3 = supervised_criterion(self.lr_model(sample[0][unsupervised_indices]), torch.tensor(y_pred_unsupervised))
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
						probs_lr = torch.nn.Softmax()(self.lr_model(sample[0]))
						loss_6 = kl_divergence(probs_lr, probs_graphical)
					else:
						loss_6= 0

					if(self.loss_func_mask[6]):
						prec_loss = precision_loss(self.theta, self.k, self.n_classes, self.qt)
					else:
						prec_loss =0

					loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_6 + loss_5 + prec_loss
					if loss != 0:
						loss.backward()
						optimizer_gm.step()
						optimizer_lr.step()

				#gm Test
				y_pred = predict_gm(self.theta, self.pi, self.l_test, self.s_test, self.k, self.n_classes, self.continuous_mask, self.qc)
				if self.use_accuracy_score:
					gm_acc = score(self.y_test, y_pred)
				else:
					gm_acc = score(self.y_test, y_pred, average = self.metric_avg)
				gm_prec = prec_score(self.y_test, y_pred, average = self.metric_avg)
				gm_recall = recall_score(self.y_test, y_pred, average = self.metric_avg)

				#gm Validation
				y_pred = predict_gm(self.theta, self.pi, self.l_valid, self.s_valid, self.k, self.n_classes, self.continuous_mask, self.qc)
				if self.use_accuracy_score:
					gm_valid_acc = score(self.y_valid, y_pred)
				else:
					gm_valid_acc = score(self.y_valid, y_pred, average = self.metric_avg)

				#LR Test
				probs = torch.nn.Softmax()(self.lr_model(self.x_test))
				y_pred = np.argmax(probs.detach().numpy(), 1)
				if self.use_accuracy_score:
					lr_acc =score(self.y_test, y_pred)
				else:
					lr_acc =score(self.y_test, y_pred, average = self.metric_avg)
				lr_prec = prec_score(self.y_test, y_pred, average = self.metric_avg)
				lr_recall = recall_score(self.y_test, y_pred, average = self.metric_avg)

				#LR Validation
				probs = torch.nn.Softmax()(self.lr_model(self.x_valid))
				y_pred = np.argmax(probs.detach().numpy(), 1)
				if self.use_accuracy_score:
					lr_valid_acc = score(self.y_valid, y_pred)
				else:
					lr_valid_acc = score(self.y_valid, y_pred, average = self.metric_avg)

				if path != None:
					file.write("Run: {}\tEpoch: {}\tgm_valid_acc: {}\tlr_valid_acc: {}".format(run, epoch, gm_valid_acc, lr_valid_acc))
					if epoch % 10 == 0:
						file.write("Run: {}\tEpoch: {}\tgm_test_acc: {}\tlr_test_acc: {}".format(run, epoch, gm_acc, lr_acc))

				if epoch > self.start_len and gm_valid_acc >= best_score_gm_val and gm_valid_acc >= best_score_lr_val:
					if gm_valid_acc == best_score_gm_val or gm_valid_acc == best_score_lr_val:
						if best_score_gm < gm_acc or best_score_lr < lr_acc:
							best_epoch_lr = epoch
							best_score_lr_val = lr_valid_acc
							best_score_lr = lr_acc

							best_epoch_gm = epoch
							best_score_gm_val = gm_valid_acc
							best_score_gm = gm_acc

							best_score_lr_prec = lr_prec
							best_score_lr_recall  = lr_recall
							best_score_gm_prec = gm_prec
							best_score_gm_recall  = gm_recall
					else:
						best_epoch_lr = epoch
						best_score_lr_val = lr_valid_acc
						best_score_lr = lr_acc

						best_epoch_gm = epoch
						best_score_gm_val = gm_valid_acc
						best_score_gm = gm_acc

						best_score_lr_prec = lr_prec
						best_score_lr_recall  = lr_recall
						best_score_gm_prec = gm_prec
						best_score_gm_recall  = gm_recall
						stop_early_lr = []
						stop_early_gm = []

				if epoch > self.start_len and lr_valid_acc >= best_score_lr_val and lr_valid_acc >= best_score_gm_val:
					if lr_valid_acc == best_score_lr_val or lr_valid_acc == best_score_gm_val:
						if best_score_lr < lr_acc or best_score_gm < gm_acc:
							
							best_epoch_lr = epoch
							best_score_lr_val = lr_valid_acc
							best_score_lr = lr_acc

							best_epoch_gm = epoch
							best_score_gm_val = gm_valid_acc
							best_score_gm = gm_acc

							best_score_lr_prec = lr_prec
							best_score_lr_recall  = lr_recall
							best_score_gm_prec = gm_prec
							best_score_gm_recall  = gm_recall
					else:
						best_epoch_lr = epoch
						best_score_lr_val = lr_valid_acc
						best_score_lr = lr_acc

						best_epoch_gm = epoch
						best_score_gm_val = gm_valid_acc
						best_score_gm = gm_acc
						best_score_lr_prec = lr_prec

						best_score_lr_recall  = lr_recall
						best_score_gm_prec = gm_prec
						best_score_gm_recall  = gm_recall
						stop_early_lr = []
						stop_early_gm = []

				if len(stop_early_lr) > self.stop_len and len(stop_early_gm) > self.stop_len and (all(best_score_lr_val >= k for k in stop_early_lr) or \
				all(best_score_gm_val >= k for k in stop_early_gm)):
					#print('Early Stopping at', best_epoch_gm, best_score_gm, best_score_lr)
					#print('Validation score Early Stopping at', best_epoch_gm, best_score_lr_val, best_score_gm_val)
					break
				else:
					stop_early_lr.append(lr_valid_acc)
					stop_early_gm.append(gm_valid_acc)

				#epoch for loop ended

			# print('Best Epoch LR', best_epoch_lr)
			# print('Best Epoch GM', best_epoch_gm)
			# print("Run \t", run, "Epoch, GM, LR \t", best_score_gm, best_score_lr)
			# print("Run \t", run, "GM Val, LR Val \t", best_score_gm_val, best_score_lr_val)

			final_score_gm.append(best_score_gm)
			final_score_gm_prec.append(best_score_gm_prec)
			final_score_gm_recall.append(best_score_gm_recall)
			final_score_gm_val.append(best_score_gm_val)
			final_score_lr.append(best_score_lr)
			final_score_lr_prec.append(best_score_lr_prec)
			final_score_lr_recall.append(best_score_lr_recall)
			final_score_lr_val.append(best_score_lr_val)

		#run for loop ended

		y_pred = predict_gm(self.theta, self.pi, self.l_test, self.s_test, self.k, self.n_classes, self.continuous_mask, self.qc)
		if self.use_accuracy_score:
			gm_acc = score(self.y_test, y_pred)
		else:
			gm_acc = score(self.y_test, y_pred, average = self.metric_avg)

		probs = torch.nn.Softmax()(self.lr_model(self.x_test))
		y_pred = np.argmax(probs.detach().numpy(), 1)
		if self.use_accuracy_score:
			lr_acc =score(self.y_test, y_pred)
		else:
			lr_acc =score(self.y_test, y_pred, average = self.metric_avg)

		if path != None:
			file.write("Training is done. gm_test_acc: {}\tlr_test_acc: {}".format(gm_acc, lr_acc))
			# print("===================================================")
			# print("TEST Averaged scores are for LR", np.mean(final_score_lr))
			# print("TEST Precision average scores are for LR", np.mean(final_score_lr_prec))
			# print("TEST Recall average scores are for LR", np.mean(final_score_lr_recall))
			# print("===================================================")
			# print("TEST Averaged scores are for GM",  np.mean(final_score_gm))
			# print("TEST Precision average scores are for GM", np.mean(final_score_gm_prec))
			# print("TEST Recall average scores are for GM", np.mean(final_score_gm_recall))
			# print("===================================================")
			# print("VALIDATION Averaged scores are for GM,LR", np.mean(final_score_gm_val), np.mean(final_score_lr_val))
			# print("TEST STD  are for GM,LR", np.std(final_score_gm), np.std(final_score_lr))
			# print("VALIDATION STD  are for GM,LR", np.std(final_score_gm_val), np.std(final_score_lr_val))
			file.close()

		if return_gm:
			return predict_gm(self.theta, self.pi, self.l_unsup, self.s_unsup, self.k, self.n_classes, self.continuous_mask, self.qc),\
			 np.argmax((torch.nn.Softmax()(self.lr_model(self.x_unsup))).detach().numpy(), 1)
		else:
			return np.argmax((torch.nn.Softmax()(self.lr_model(self.x_unsup))).detach().numpy(), 1)

	def predict_cage(self, path_test):
		'''
			Used to find the predicted labels based on the trained parameters of graphical model(CAGE)

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

	def predict_feature_model(self, path_test):
		'''
			Used to find the predicted labels based on the trained parameters of feature model

		Args:
			path_test: Path to the pickle file containing test data set
		Return:
			numpy.ndarray of shape (num_instances,) which are predicted labels
			[Note: no aggregration/algorithm-running will be done using the current input]
		'''
		data = get_data(path_test)
		x_test = data[0]
		assert x_test.shape[1] == self.n_features

		return np.argmax((torch.nn.Softmax()(self.lr_model(x_test))).detach().numpy(), 1)