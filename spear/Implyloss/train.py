# from .my_data_types import f_d, f_d_U
# from .my_utils import *
# import .my_utils, my_data_types
from .data_types import f_d, f_d_U
from .utils import *

# import metrics_utils
import json
import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pickle
import sys,os
from sklearn.metrics import precision_recall_fscore_support
from snorkel.labeling import labeling_function
from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling.model import LabelModel
from sklearn.metrics import precision_recall_fscore_support
# from snorkel_utils import conv_l_to_lsnork

# All training methods for HLS
class HLSTrain():
	'''
	Func Desc:
	This Class is designed to train the HLS model using the Implyloss Algorithm
	'''
	def __init__(self, hls, 
		f_d_metrics_pickle, f_d_U_metrics_pickle, f_d_adam_lr, f_d_U_adam_lr, early_stopping_p, f_d_primary_metric, mode, data_dir):
		'''
		Func Desc:
		Initializes the class member variables using the arguments provided

		Input:
		self
		hls - the hls model

		Sets:
		hls
		f_d_metrics_pickle
		'''
		self.hls = hls
		# self.config = config
		self.f_d_metrics_pickle = f_d_metrics_pickle #file path where metrics of trained model are stored
		self.f_d_U_metrics_pickle = f_d_U_metrics_pickle #file path where metrics of trained model are stored
		self.f_d_adam_lr = f_d_adam_lr
		self.f_d_U_adam_lr = f_d_U_adam_lr
		self.early_stopping_p = early_stopping_p
		self.f_d_primary_metric = f_d_primary_metric
		self.mode = mode
		self.data_dir = data_dir
		self.init_metrics()
		self.make_f_summary_ops()

	def make_f_summary_ops(self):
		'''
		Func Desc:
		make the summary of all the essential parameters of f_network

		Input:
		Self

		Summarizes:
		f_d_loss_ph
		f_d_loss
		f_d_f1_score_ph
		f_d_f1_score
		f_d_accuracy_ph
		f_d_accuracy
		f_d_avg_f1_score_ph
		f_d_avg_f1_score
		f_d_summaries

		'''
		with tf.name_scope('f_summaries'):
			self.f_d_loss_ph = tf.placeholder(tf.float32, shape=None, name='f_d_loss_placeholder')
			self.f_d_loss = tf.summary.scalar('f_d_loss', self.f_d_loss_ph)

			self.f_d_f1_score_ph = tf.placeholder(tf.float32, shape=None, name='f_d_f1_score_placeholder')
			self.f_d_f1_score = tf.summary.scalar('f_d_f1_score_1', self.f_d_f1_score_ph)

			self.f_d_accuracy_ph = tf.placeholder(tf.float32, shape=None, name='f_d_accuracy_placeholder')
			self.f_d_accuracy = tf.summary.scalar('f_d_accuracy', self.f_d_accuracy_ph)

			self.f_d_avg_f1_score_ph = tf.placeholder(tf.float32, shape=None, name='f_d_avg_f1_score_placeholder')
			self.f_d_avg_f1_score = tf.summary.scalar('f_d_avg_f1_score', self.f_d_f1_score_ph)

			self.f_d_summaries = tf.summary.merge([self.f_d_loss, self.f_d_f1_score,
					self.f_d_accuracy, self.f_d_avg_f1_score])

	def report_f_d_perfs_to_tensorboard(self, f_d_loss, metrics_dict, global_step):
		'''
		Func Desc:
		report the f_d_performance to tensorboard

		Input:
		self
		f_d_loss
		metrics_dict
		global_step

		Output:

		'''
		print('Reporting f_d metrics to tensorboard')
		summ = self.hls.sess.run(self.f_d_summaries, feed_dict={
			self.f_d_loss_ph: f_d_loss,
			self.f_d_f1_score_ph: metrics_dict['f1_score_1'],
			self.f_d_avg_f1_score_ph: metrics_dict['avg_f1_score'],
			self.f_d_accuracy_ph: metrics_dict['accuracy']
			})
		self.hls.writer.add_summary(summ, global_step=global_step)

	def train_f_on_d(self, datafeeder, num_epochs):
		'''
		Func Desc:
		trains the f_network (classification network) on labelled data

		Input:
		self
		datafeeder - datafeeder object
		num_epochs - number of epochs for training

		Output:
		'''
		train_op = self.hls.f_d_train_op
		loss_op = self.hls.f_d_loss

		sess = self.hls.sess
		total_batch = datafeeder.get_batches_per_epoch(f_d)
		batch_size = datafeeder.get_batch_size(f_d)
		with sess.as_default():
			# Training cycle
			print("Optimization started for f_d!")
			print("Batch size: %d!" % batch_size)
			print("Batches per epoch : %d!" % total_batch)
			print("Number of epochs: %d!" % num_epochs)

			metrics_dict = {} #{'config': self.config}

			best_saver_f_d = self.hls.best_savers.get_best_saver(f_d)
			patience = 0
			for epoch in range(num_epochs):
				avg_cost = 0.
				for i in range(total_batch):
					batch_x, batch_y = datafeeder.get_f_d_next_batch()
					feed_dict = {
							self.hls.f_d_adam_lr: self.f_d_adam_lr,
							self.hls.f_x: batch_x,
							self.hls.f_d_labels: batch_y
							}

					merge_dict_a_into_b(self.hls.dropout_train_dict,feed_dict)

					# Run optimization op (backprop) and cost op (to get loss value)
					_, c, f_d_global_step, global_step = sess.run([train_op, loss_op, 
														 self.hls.f_d_global_step, 
														 self.hls.global_step], 
														 feed_dict=feed_dict)

					global_epoch = f_d_global_step / total_batch
					# Compute average loss
					avg_cost += c / total_batch
					cost1 = (avg_cost * total_batch ) / (i + 1)                
				# Compute and report metrics, update checkpoints after each epoch
				print("\n========== epoch : {} ============\n".format(epoch))
				print("cost: {}\n".format(cost1))
				print("patience: {}\n".format(patience))
				precision, recall, f1_score, support = self.hls.test.test_f(datafeeder)
				self.compute_f_d_metrics(metrics_dict, precision, recall, f1_score, support,
						global_epoch, f_d_global_step)
				print("\nmetrics_dict: ", metrics_dict)
				print()
				self.report_f_d_perfs_to_tensorboard(cost1, metrics_dict, global_step)
				did_improve = self.maybe_save_metrics_dict(f_d, metrics_dict)
				if did_improve:
					patience = 0 #rest patience if primary metric improved
				else:
					patience += 1
					if patience > self.early_stopping_p:
						print("bye! stopping early!......")
						break
				# Save checkpoint
				print()
				self.hls.mru_saver.save(global_step)
				print()
				best_saver_f_d.save_if_best(metrics_dict[self.f_d_primary_metric])
				print()
			print("Optimization Finished for f_d!")

	def train_f_on_d_U(self, datafeeder, num_epochs, loss_type):
		'''
		Func Desc:
		trains the f_network (classification network) on labelled amd unlabelled data

		Input:
		self
		datafeeder - datafeeder object
		num_epochs - number of epochs for training
		loss_type - different available losses

		Output:
		'''
		sess = self.hls.sess

		total_batch = datafeeder.get_batches_per_epoch(f_d_U)
		batch_size = datafeeder.get_batch_size(f_d_U)
		
		if loss_type == 'pure-likelihood':
			train_op = self.hls.f_d_U_pure_likelihood_op
			loss_op = self.hls.f_d_U_pure_likelihood_loss
		elif loss_type == 'implication':
			train_op = self.hls.f_d_U_implication_op
			loss_op = self.hls.f_d_U_implication_loss
		elif loss_type == 'pr_loss':
			train_op = self.hls.pr_train_op
			loss_op = self.hls.pr_loss
		elif loss_type == 'gcross':
			train_op = self.hls.gcross_train_op
			loss_op = self.hls.gcross_loss
		elif loss_type == 'gcross_snorkel':
			train_op = self.hls.snork_gcross_train_op
			loss_op = self.hls.snork_gcross_loss
		elif loss_type == 'learn2reweight':
			train_op = self.hls.l2r_train_op
			loss_op = self.hls.l2r_loss
		elif loss_type == 'label_snorkel':
			train_op = self.hls.label_snorkel_train_op
			loss_op = self.hls.label_snorkel_loss
		elif loss_type == 'pure_snorkel':
			train_op = self.hls.pure_snorkel_train_op
			loss_op = self.hls.pure_snorkel_loss        
		else:
			raise ValueError('Invalid loss type %s' % loss_type)
		
		best_saver_f_d_U = self.hls.best_savers.get_best_saver(f_d_U)
		metrics_dict = {} #{'config': self.config}

		if 'label_snorkel' == self.mode or 'pure_snorkel' == self.mode or 'gcross_snorkel' == self.mode:
		    label_model = LabelModel(cardinality=self.hls.num_classes, verbose=True)
		    if os.path.isfile(os.path.join(self.data_dir,"saved_label_model")):
		        label_model = label_model.load(os.path.join(self.data_dir,"saved_label_model"))
		    else:
		        print("LABEL MODEL NOT SAVED")
		        exit()
		if 'gcross' in self.mode or 'learn2reweight' in self.mode:
		    majority_model = MajorityLabelVoter(cardinality=self.hls.num_classes)

		with sess.as_default():
			print("Optimization started for f_d_U with %s loss!" % loss_type)
			print("Batch size: %d!" % batch_size)
			print("Batches per epoch : %d!" % total_batch)
			print("Number of epochs: %d!" % num_epochs)
			# Training cycle
			iteration = 0
			global_step = 0
			patience = 0
			for epoch in range(num_epochs):
				avg_epoch_cost = 0.

				for i in range(total_batch):
					batch_x, batch_l, batch_m, batch_L, batch_d, batch_r =\
							datafeeder.get_f_d_U_next_batch()

					feed_dict={
							self.hls.f_d_U_adam_lr: self.f_d_U_adam_lr,
							self.hls.f_d_U_x: batch_x,
							self.hls.f_d_U_l : batch_l,
							self.hls.f_d_U_m : batch_m, 
							self.hls.f_d_U_L : batch_L,
							self.hls.f_d_U_d : batch_d,
							self.hls.f_d_U_r : batch_r
							}

					batch_lsnork = conv_l_to_lsnork(batch_l,batch_m)

					if 'label_snorkel' == self.mode or 'pure_snorkel' == self.mode or 'gcross_snorkel' == self.mode:                        
						batch_snork_L = label_model.predict_proba(L=batch_lsnork) #snorkel_probs
						feed_dict[self.hls.f_d_U_snork_L] = batch_snork_L

					if 'gcross' == self.mode or 'learn2reweight' == self.mode:
						batch_snork_L = majority_model.predict(L=batch_lsnork) #majority votes
						batch_snork_L = np.eye(self.hls.num_classes)[batch_snork_L] #one hot rep
						feed_dict[self.hls.f_d_U_snork_L] = batch_snork_L                        

					merge_dict_a_into_b(self.hls.dropout_train_dict, feed_dict)
					# Run optimization op (backprop) and cost op (to get loss value)
					_, cost, num_d, f_d_U_global_step = sess.run([
							train_op,
							loss_op,
							self.hls.f_d_U_num_d,
							self.hls.f_d_U_global_step],
							feed_dict=feed_dict
							)

					global_epoch = f_d_U_global_step / total_batch
					# This assertion is valid only if true U labels are available but not being used such as for
					# synthetic data.
					assert np.all(batch_L <= self.hls.num_classes)

					avg_epoch_cost += cost / total_batch
					cost1 = (avg_epoch_cost * total_batch ) / (i + 1)
					global_step += 1

				# Compute and report metrics, update checkpoints after each epoch
				print("\n========== epoch : {} ============\n".format(epoch))
				print("cost: {}\n".format(cost1))
				print("patience: {}\n".format(patience))
				precision, recall, f1_score, support = self.hls.test.test_f(datafeeder)
				self.compute_f_d_metrics(metrics_dict, precision, recall, f1_score, support,
						global_epoch, f_d_U_global_step)
				print("\nmetrics_dict: ", metrics_dict)
				print()
				self.report_f_d_perfs_to_tensorboard(cost1, metrics_dict, global_step)
				did_improve = self.maybe_save_metrics_dict(f_d_U, metrics_dict)
				if did_improve:
					patience = 0 #rest patience if primary metric improved
				else:
					patience += 1
					if patience > self.early_stopping_p:
						print("bye! stopping early!......")
						break
				# Save checkpoint
				print()
				self.hls.mru_saver.save(global_step)
				print()
				best_saver_f_d_U.save_if_best(metrics_dict[self.f_d_primary_metric])
				print()
				global_step += 1
			print("Optimization Finished for f_d_U!")


	def init_metrics(self):
		'''
		Func desc:
		initialize the metrics

		Input:
		self

		Output:
		
		'''
		self.metrics_file = {
				f_d: self.f_d_metrics_pickle,
				f_d_U: self.f_d_U_metrics_pickle,
				}
	
		self.best_metric = {}
		self.best_metrics_dict = {}
		for run_type in [f_d, f_d_U]:
			try:
				with open(self.metrics_file[run_type], 'rb') as f:
					metrics_dict = pickle.load(f)
				self.best_metric[run_type] = self.get_metric(run_type, metrics_dict)
				self.best_metrics_dict[run_type] = metrics_dict
				print('Found prev best metric for run type %s: %.3f' % (run_type, self.best_metric[run_type]))
				print('best metrics dict: ', self.best_metrics_dict[run_type])
			except FileNotFoundError as e:
				print(str(e))
				self.best_metric[run_type] = 0.
				self.best_metrics_dict[run_type] = {}
				print('Did not find prev best metric for run type %s. Setting to zero.' % (run_type))

	def get_metric(self, run_type, metrics_dict):
		'''
		Func desc:
		get the metrics

		Input:
		self
		run_type
		metrics_dict

		Output:
		the required metrics_dict
		'''
		return metrics_dict[self.f_d_primary_metric]

	def save_metrics(self, run_type, metrics_dict):
		'''
		Func desc:
		save the metrics

		Input:
		self
		run_type
		metrics_dict

		Prints:
		The saved metric file
		
		'''
		with open(self.metrics_file[run_type], 'wb') as f:
			pickle.dump(metrics_dict, f)
		print('\ndumped metrics dict to file: ', self.metrics_file[run_type])

	def maybe_save_metrics_dict(self, run_type, metrics_dict):
		'''
		Func desc:
		save the metric if it is the best till now

		Input:
		self
		run_type
		metrics_dict

		Output:
		True or False denoting whether the current metric is saved or not

		Prints:
		The saved metric file

		'''
		metric = self.get_metric(run_type, metrics_dict)
		if self.best_metric[run_type] < metric:
			self.best_metric[run_type] = metric
			self.best_metrics_dict[run_type] = metrics_dict
			self.save_metrics(run_type, metrics_dict)
			return True
		else:
			print('Not saving metrics dict. Best metric value is', self.best_metric[run_type],
					'Current is:', metric)
			return False

	def compute_f_d_metrics(self, metrics_dict, precision, recall, f1_score, support, global_epoch, f_d_global_step):
		'''
		Func desc:
		compute the f_d metrics

		input:
		self
		metrics_dict
		precision
		recall
		f1_score
		support
		global_epoch
		f_d_global_step

		output:
		void

		evaluates:
		metrics_dict, accuracy
		'''
		# Class = 1 metrics
		if len(f1_score) == 1:
			metrics_dict['f1_score_1'] = 0
			metrics_dict['precision_1'] = 0
			metrics_dict['recall_1'] = 0
			metrics_dict['support_1'] = 0
		else:
			metrics_dict['f1_score_1'] = f1_score[1]
			metrics_dict['precision_1'] = precision[1]
			metrics_dict['recall_1'] = recall[1]
			metrics_dict['support_1'] = support[1]       

		# All classes metrics
		metrics_dict['f1_score'] = f1_score
		metrics_dict['precision'] = precision
		metrics_dict['recall'] = recall
		metrics_dict['support'] = support

		# Aggregate metrics
		metrics_dict['avg_f1_score'] = sum(f1_score) / len(f1_score)
		metrics_dict['avg_precision'] = sum(precision) / len(precision)
		metrics_dict['avg_recall'] = sum(recall) / len(recall)
		
		# Accuracy is recall weighted by support
		accuracy = sum(recall * support) / sum(support)
		metrics_dict['accuracy'] = accuracy
		# Extra stats
		metrics_dict['epoch'] = global_epoch
		metrics_dict['f_d_global_step'] = f_d_global_step

