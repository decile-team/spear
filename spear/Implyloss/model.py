# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import functools
import math
import numpy as np
import sys,os

# from .my_checkpoints import MRUCheckpoint, CheckpointsFactory
# from .my_data_types import *
# # import my_gen_cross_entropy_utils as gcross_utils
# from .my_gen_cross_entropy_utils import *
# from .my_gen_cross_entropy_utils import *
# # import my_pr_utils
# from .my_pr_utils import *
# from .my_test import HLSTest
# from .my_train import HLSTrain
# from .my_utils import print_tf_global_variables, updated_theta_copy

from .checkpoints import MRUCheckpoint, CheckpointsFactory
from .data_types import *
# import gen_cross_entropy_utils as gcross_utils
from .gen_cross_entropy_utils import *
from .gen_cross_entropy_utils import *
# import pr_utils
from .pr_utils import *
from .test import HLSTest
from .train import HLSTrain
from .utils import print_tf_global_variables, updated_theta_copy



class HighLevelSupervisionNetwork:
	# Parameters
	display_step = 1
	'''
	Class Desc:
	Initialize HLS with number of input features, number of classes, number of rules and the f and the w network.
	f network is the classification network (P_{theta})
	w network is the rule network (P_{j,phi})
	'''
	def __init__(self, num_features, num_classes, num_rules,
			num_rules_to_train, rule_classes,
			w_network, f_network, 
			f_d_epochs, f_d_U_epochs, f_d_adam_lr, f_d_U_adam_lr, dropout_keep_prob, 
			f_d_metrics_pickle, f_d_U_metrics_pickle, early_stopping_p, f_d_primary_metric, mode, data_dir, 
			tensorboard_dir, checkpoint_dir, checkpoint_load_mode, gamma, lamda, raw_d_x=None, raw_d_L=None):
		'''
		Func Desc:
		initializes the class member variables with the provided arguments

		Input:
		self
		num_features
		num_classes
		num_rules
		num_rules_to_train
		rule_classes
		w_network
		f_network
		raw_d_x (default = None)
		raw_d_L (default = None)

		Output:

		'''
		
		# Modules for testing/training

		self.mode = mode
		self.gamma = gamma
		self.lamda = lamda
		self.raw_d_x = raw_d_x
		self.raw_d_L = raw_d_L
		self.rule_classes_list = rule_classes
		self.rule_classes = tf.convert_to_tensor(rule_classes)
		self.num_features = num_features
		self.num_classes = num_classes
		self.num_rules = num_rules
		self.num_rules_to_train = num_rules_to_train
		self.w_var_scope = 'w_network'
		self.f_var_scope = 'f_network'
		self.w_network = functools.partial(w_network, self.w_var_scope,
				self.num_rules)
		self.f_network = functools.partial(f_network, self.f_var_scope)
		self.parse_params(f_d_epochs, f_d_U_epochs, f_d_adam_lr, f_d_U_adam_lr)

		# Save global step for each different kind of run
		self.global_steps = {}
		# Global global step 
		self.global_step = tf.train.get_or_create_global_step()

		# Create the compute graphs
		# dropout rate used in f and w networks
		self.dropout_keep_prob = tf.placeholder(tf.float32,name="keep_prob")
		self.dropout_train_dict = {self.dropout_keep_prob: dropout_keep_prob}
		self.dropout_test_dict = {self.dropout_keep_prob: 1.0}
		
		self.train = HLSTrain(self, f_d_metrics_pickle, 
									f_d_U_metrics_pickle, 
									f_d_adam_lr, 
									f_d_U_adam_lr, 
									early_stopping_p, 
									f_d_primary_metric, 
									mode, 
									data_dir)
		self.test = HLSTest(self)
		
		self.make_f_d_train_ops()
		self.make_f_d_U_train_ops()

		# Print all global variables
		print_tf_global_variables()

		# Initialize all variables
		self.init = tf.global_variables_initializer()
		sess_config = tf.ConfigProto()
		sess_config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=sess_config)

		self.writer = tf.summary.FileWriter(tensorboard_dir, self.sess.graph)
		# Now that all variables and the session is created, create a
		# checkpoint saver. We use a single saver for all variables
		self.mru_saver = MRUCheckpoint(checkpoint_dir, self.sess, tf.global_variables())
		self.best_savers = CheckpointsFactory(self.sess, self.global_steps)
		feed_dict = {}
		self.sess.run(self.init, feed_dict=feed_dict)
		if checkpoint_load_mode == 'mru':
			self.restored = self.mru_saver.restore_if_checkpoint_exists()
		else:
			saver = self.best_savers.get_best_saver(checkpoint_load_mode)
			self.restored = saver.restore_best_checkpoint_if_exists()
			if not self.restored:
				raise ValueError('Asked to restore best checkpoint of %s but not previously checkpointed' %
						checkpoint_load_mode)

	def parse_params(self, f_d_epochs, f_d_U_epochs, f_d_adam_lr, f_d_U_adam_lr):
		'''
		Func Desc:
		Parses the given parameters 

		Input:
		self 

		Sets:
		f_d_epochs
		f_d_U_epochs
		initial_f_d_adam_lr
		initial_f_d_U_adam_lr
		'''
		self.f_d_epochs     =   f_d_epochs
		self.f_d_U_epochs   = f_d_U_epochs
		self.initial_f_d_adam_lr = f_d_adam_lr
		self.initial_f_d_U_adam_lr = f_d_U_adam_lr

	# Create the train op for training with d only
	def make_f_d_train_ops(self):
		'''
		Func Desc:
		create the train_ops based on labelled data only

		Input:
		self

		Output:

		'''
		self.f_d_global_step = tf.Variable(0, trainable=False, name='f_d_global_step')
		inc_f_d_global_step = tf.assign_add(self.f_d_global_step, 1)
		self.global_steps[f_d] = self.f_d_global_step
		self.f_d_adam_lr = tf.placeholder(tf.float32,name='f_d_U_adam_lr')

		# [batch size, features]
		self.f_x = tf.placeholder(tf.float32, shape=[None, self.num_features],
				name='f_d_x')
		self.f_d_labels = tf.placeholder(tf.float32, shape=[None,
			self.num_classes], name='f_d_labels')

		f_dict = {'x': self.f_x, 'labels': self.f_d_labels}

		self.f_d_logits = self.f_network(f_dict, self.num_classes,
				dropout_keep_prob=self.dropout_keep_prob)

		self.f_d_probs = tf.math.softmax(self.f_d_logits, axis=-1)
		self.f_d_preds = tf.argmax(self.f_d_probs, axis=-1)        

		
		model_loss = tf.nn.softmax_cross_entropy_with_logits(
			labels=self.f_d_labels, logits=self.f_d_logits)

		self.f_d_loss = tf.reduce_mean(model_loss)

		self.f_d_optimizer = tf.train.AdamOptimizer(
				learning_rate=self.f_d_adam_lr,
				name='f_d_Adam')

		with tf.control_dependencies([inc_f_d_global_step]):
			self.f_d_train_op = self.f_d_optimizer.minimize(self.f_d_loss, global_step=self.global_step)

	def make_f_d_U_train_ops(self):
		self.f_d_U_global_step = tf.Variable(0, trainable=False, name='f_d_U_global_step')
		inc_f_d_U_global_step = tf.assign_add(self.f_d_U_global_step, 1)
		self.global_steps[f_d_U] = self.f_d_U_global_step
		'''
		Func desc:
		make_f_d_U_train_ops i.e. training ops by combining labelled and unlabelled data, compute the training objective and aim to minimize the loss function using the adam optimizer

		Input:
		self object

		Sets:
		* x : feature representation of instance
			- shape : [batch_size, num_features]

		* l : Labels assigned by rules
			- shape [batch_size, num_rules]
			- l[i][j] provides the class label provided by jth rule on ith instance
			- if jth rule does not fire on ith instance, then l[i][j] = num_classes (convention)
			- in snorkel, convention is to keep l[i][j] = -1, if jth rule doesn't cover ith instance
			- class labels belong to {0, 1, 2, .. num_classes-1}

		* m : Rule coverage mask
			- A binary matrix of shape [batch_size, num_rules]
			- m[i][j] = 1 if jth rule cover ith instance
			- m[i][j] = 0 otherwise

		* L : Instance labels
			- shape : [batch_size, 1]
			- L[i] = label of ith instance, if label is available i.e. if instance is from labeled set d
			- Else, L[i] = num_clases if instances comes from the unlabeled set U
			- class labels belong to {0, 1, 2, .. num_classes-1}

		* d : binary matrix of shape [batch_size, 1]
			- d[i] = 1 if instance belongs to labeled data (d), d[i]=0 otherwise
			- d[i]=1 for all instances is from d_processed.p
			- d[i]=0 for all instances in other 3 pickles {U,validation,test}_processed.p
		
		* r : A binary matrix of shape [batch_size, num_rules]
			- r[i][j]=1 if jth rule was associated with ith instance
			- Highly sparse matrix
			- r is a 0 matrix for all the pickles except d_processed.p
			- Note that this is different from rule coverage matrix "m"
			- This matrix defines the rule,example pairs provided as supervision 

		* s : A similarity measure matrix shape [batch_size, num_rules]
			- s[i][j] is in [0,1]

		* n : A vector of size [num_rules,]
			- Mask for s (denotes whether particular rule is continuous or discrete)

		* k : a vector of size [num_rules,]
			- #LF classes ie., what class each LF correspond to, range: 0 to num_classes-1

		Computes:
		weights, w_logits of rule network ([batch_size, num_rules]) - Used to train P_j_phi(r_j/x_i) i.e. whether rij = 1 for the ith instance and jth rule 
		f_logits of the classification network - Used to train P_j_theta(l_j/x_i) i.e. the probability of ith instance belonging to jth class
		LL_phi term
		LL_theta term
		Training objective term
		Minimum loss using adam optimizer
		'''

		self.f_d_U_adam_lr = tf.placeholder(tf.float32,name='f_d_U_adam_lr')
		self.f_d_U_x = tf.placeholder(
				tf.float32,
				shape=[None, self.num_features],
				name='f_d_U_x')
		# l
		self.f_d_U_l = tf.placeholder(
				tf.int32,
				shape=[None, self.num_rules],
				name='f_d_U_l')
		# m 
		self.f_d_U_m = tf.placeholder(tf.float32, shape=[None,
			self.num_rules], name='f_d_U_m')
		# L
		L = self.f_d_U_L = tf.placeholder(tf.int32, shape=[None, 1], name='f_d_U_L')
		# d
		d = self.f_d_U_d = tf.placeholder(tf.float32, shape=[None, 1], name='f_d_U_d')
		L = tf.squeeze(L)
		d = tf.squeeze(d)

		r = self.f_d_U_r = tf.placeholder(tf.float32, shape=[None,self.num_rules], name='f_d_U_r')

		#weights: [batch_size, num_rules]
		#w_logits: [batch_size, num_rules]
		weights, w_logits = self.get_weights_and_logits_f_d_U(self.f_d_U_x)
		self.f_d_U_weights = weights
		self.f_d_U_num_d = tf.reduce_sum(d) #number of labeled instances in a batch

		# w_network computation is done. Now run f_network to get logits for
		# this batch
		f_dict = {'x': self.f_d_U_x}
		f_logits = self.f_network(f_dict, self.num_classes, reuse=True, 
								dropout_keep_prob=self.dropout_keep_prob)
		self.f_d_U_probs = tf.math.softmax(f_logits, axis=-1)
		self.f_d_U_preds = tf.argmax(self.f_d_U_probs, axis=-1)
		self.joint_f_w_score = self.joint_scores_from_f_and_w(self.f_d_U_weights,self.f_d_U_m,self.f_d_U_probs)

		# Do this so that the cross-entropy does not blow for data from U
		# The actual value of cross-entropy for U does not matter since it
		# will be multiplied by 0 anyway.        
		L = L % self.num_classes

		# Ok now compute the loss LL_theta which is on d data
		L_one_hot = tf.one_hot(L, self.num_classes)
		LL_theta = tf.nn.softmax_cross_entropy_with_logits(logits=f_logits,
				labels=L_one_hot)
		LL_theta = d * LL_theta
		LL_theta = tf.reduce_mean(LL_theta) # loss of f network on labeled data d

		LL_theta = LL_theta # loss of f network on labeled data d
								 # first term in eqn 5 (LL(\theta))


		# LL(\phi) term
		LL_phi = self.compute_LL_phi(w_logits=w_logits, 
														  weights=self.f_d_U_weights,
														  l=self.f_d_U_l, 
														  m=self.f_d_U_m,
														  L=L,
														  d=d,
														  r=self.f_d_U_r)
		

		f_cross_training_optimizer = tf.train.AdamOptimizer(
				learning_rate=self.f_d_U_adam_lr,
				name='f_d_U_Adam')

		training_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)


		if 'implication' == self.mode:
			implication_loss = self.implication_loss(weights=self.f_d_U_weights,
													f_probs=self.f_d_U_probs,
													m=self.f_d_U_m,
													rule_classes=self.rule_classes,
													num_classes=self.num_classes,
													d=d)
			
			self.f_d_U_implication_loss = LL_phi \
										+ LL_theta \
										+ self.gamma*implication_loss

			with tf.control_dependencies([inc_f_d_U_global_step,  ]):
				self.f_d_U_implication_op = f_cross_training_optimizer.minimize(
						self.f_d_U_implication_loss,
						var_list=training_var_list)

		if 'pr_loss' == self.mode:
# 			pr_loss = my_pr_utils.pr_loss(m=self.f_d_U_m,
			pr_loss_val = pr_loss(m=self.f_d_U_m,
									f_logits=f_logits, 
									w_logits=w_logits, 
									f_probs=self.f_d_U_probs,
									weights=self.f_d_U_weights,
									rule_classes=self.rule_classes,
									num_classes=self.num_classes, 
									C=0.1,
									d=d)
			self.pr_loss = LL_theta + LL_phi + self.gamma*pr_loss_val 
			with tf.control_dependencies([inc_f_d_U_global_step,  ]):
				self.pr_train_op = f_cross_training_optimizer.minimize(
									self.pr_loss,
									var_list=training_var_list)

		if 'gcross' == self.mode:
			self.f_d_U_snork_L = tf.placeholder(
					tf.float32,
					shape=[None, self.num_classes],
					name='f_d_U_snork_L')

			loss_on_d = LL_theta
# 			loss_on_U = gcross_utils.generalized_cross_entropy(f_logits,self.f_d_U_snork_L,
# 													self.lamda)
			loss_on_U = generalized_cross_entropy(f_logits,self.f_d_U_snork_L,
													self.lamda)
			self.gcross_loss = loss_on_d + self.gamma*loss_on_U
			with tf.control_dependencies([inc_f_d_U_global_step,  ]):
				self.gcross_train_op = f_cross_training_optimizer.minimize(
									self.gcross_loss,
									var_list=training_var_list)

		if 'gcross_snorkel' == self.mode:
			self.f_d_U_snork_L = tf.placeholder(
					tf.float32,
					shape=[None, self.num_classes],
					name='f_d_U_snork_L')

			loss_on_d = LL_theta
# 			loss_on_U = gcross_utils.generalized_cross_entropy(f_logits,self.f_d_U_snork_L,
# 													self.lamda)
			loss_on_U = generalized_cross_entropy(f_logits,self.f_d_U_snork_L,
													self.lamda)
			self.snork_gcross_loss = loss_on_d + self.gamma*loss_on_U
			#self.snork_gcross_loss = loss_on_d + loss_on_U
			with tf.control_dependencies([inc_f_d_U_global_step,  ]):
				self.snork_gcross_train_op = f_cross_training_optimizer.minimize(
									self.snork_gcross_loss,
									var_list=training_var_list)

		if 'label_snorkel' == self.mode or 'pure_snorkel' == self.mode:
			self.f_d_U_snork_L = tf.placeholder(
					tf.float32,
					shape=[None, self.num_classes],
					name='f_d_U_snork_L')
			loss_on_d = LL_theta
			self.pure_snorkel_loss = tf.nn.softmax_cross_entropy_with_logits(
									 labels=self.f_d_U_snork_L,logits=f_logits)
			self.pure_snorkel_loss = tf.reduce_mean(self.pure_snorkel_loss)
			self.label_snorkel_loss = loss_on_d + self.gamma*self.pure_snorkel_loss

			if 'label_snorkel' == self.mode:
				with tf.control_dependencies([inc_f_d_U_global_step,  ]):
					self.label_snorkel_train_op = f_cross_training_optimizer.minimize(
										self.label_snorkel_loss,
										var_list=training_var_list)

			if 'pure_snorkel' == self.mode:
				with tf.control_dependencies([inc_f_d_U_global_step,  ]):
					self.pure_snorkel_train_op = f_cross_training_optimizer.minimize(
										self.pure_snorkel_loss,
										var_list=training_var_list)

		if 'learn2reweight' == self.mode:
			len_raw_d_x = len(self.raw_d_x)
			raw_d_bs = min(len_raw_d_x,32)
			raw_d_x = tf.get_variable(name="raw_d_x", initializer=self.raw_d_x, trainable=False)
			raw_d_x = tf.to_float(raw_d_x)
			raw_d_L = tf.get_variable(name="raw_d_L", initializer=self.raw_d_L, trainable=False)
			raw_d_L = tf.to_int32(raw_d_L)
			#raw_d_L = tf.expand_dims(raw_d_L,1)
			batch_points = tf.random.uniform([raw_d_bs],minval=0,maxval=len_raw_d_x, dtype=tf.int32)
			one_hot_batch_points_float = tf.one_hot(batch_points,len_raw_d_x,dtype=tf.float32)
			batch_raw_d_x = tf.matmul(one_hot_batch_points_float,raw_d_x)
			one_hot_batch_points_int = tf.one_hot(batch_points,len_raw_d_x,dtype=tf.int32)
			batch_raw_d_L = tf.matmul(one_hot_batch_points_int,raw_d_L)
			batch_raw_d_L = tf.squeeze(batch_raw_d_L) 

			self.f_d_U_snork_L = tf.placeholder(
					tf.float32,
					shape=[None, self.num_classes],
					name='f_d_U_snork_L')

			# 1. initialize epsilon
			# [batch_size]
			epsilon = tf.zeros(tf.shape(self.f_d_U_x)[0])
			
			# 2. compute epsilon weighted loss (ewl) for batch
			#[batch_size, num_classes]
			f_logits = self.f_network(f_dict, self.num_classes, reuse=True,
				dropout_keep_prob=self.dropout_keep_prob)
			#[batch_size]
			unweighted_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.f_d_U_snork_L,logits=f_logits)
			weighted_loss = epsilon * unweighted_loss 
			weighted_loss = tf.reduce_sum(weighted_loss)
			
			# 3. compute grads of ewl wrt thetas
			thetas = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.f_var_scope)
			grads_thetas = tf.gradients(ys=weighted_loss,xs=thetas,stop_gradients=epsilon)

			
			# 4. update theta
			theta_hat = updated_theta_copy(
													grads=grads_thetas,
													variables=thetas,
													lr=self.lamda,
													mode=-1)

			# 5. compute unweighted loss on raw_d with updated theta (theta_hat)            
			f_dict_on_d = {'x': batch_raw_d_x}
			f_logits_on_d = self.f_network(f_dict_on_d, self.num_classes, 
											   reuse=False, ph_vars=theta_hat,
											   dropout_keep_prob=self.dropout_keep_prob)
			raw_d_L_one_hot = tf.one_hot(batch_raw_d_L,self.num_classes,dtype=tf.float32)
			unweighted_loss = tf.nn.softmax_cross_entropy_with_logits(labels=raw_d_L_one_hot,
																		  logits=f_logits_on_d)
			unweighted_loss = tf.reduce_mean(unweighted_loss)

			# 6. compute grads of unweighted loss wrt epsilons
			grad_epsilon = tf.gradients(ys=unweighted_loss,xs=epsilon,stop_gradients=thetas)[0]
			#grad_epsilon = tf.Print(grad_epsilon,[grad_epsilon],message="\n\n\n grad_epsilon \n\n\n")                

			# 7. truncate and normalize grad-epsilons to get w
			w_tilde = tf.nn.relu(-grad_epsilon)
			w_norm = w_tilde/(tf.reduce_sum(w_tilde) + 1e-25)
			#w_norm = tf.Print(w_norm,[w_norm],message="\n\n\n w_norm \n\n\n")

			# 8. Compute ewl for batch with original theta and normalized weights
			f_logits = self.f_network(f_dict,self.num_classes,reuse=True,
									  dropout_keep_prob=self.dropout_keep_prob)
			unweighted_loss = tf.nn.softmax_cross_entropy_with_logits(
									 labels=self.f_d_U_snork_L,logits=f_logits)
			w_norm = tf.stop_gradient(w_norm)
			weighted_loss = w_norm * unweighted_loss
			weighted_loss = tf.reduce_sum(weighted_loss)
			self.l2r_loss = weighted_loss
			with tf.control_dependencies([inc_f_d_U_global_step,  ]):
				# 9. Compute grads of ewl wrt to original theta to obtain Update theta operation
				self.l2r_train_op = f_cross_training_optimizer.minimize(
									self.l2r_loss,
									var_list=thetas)

	def compute_LL_phi(self, w_logits, weights, l, m, L, d, r):
		'''
		Func desc: 
		Computes the LL_phi term coming in the training objective

		Input:
		self object
		w_logits([batch_size, num_rules]) - 
		weights([batch_size, num_rules]) - the weights matrix corresponding to rule network(w_network) in the algorithm
		l([batch_size, num_rules]) - labels assigned by the rules
		m([batch_size, num_rules]) - the rule coverage matrix where m_ij = 1 if jth rule covers ith instance 
		L([batch_size, 1]) - L_i = 1 if the ith instance has already a label assigned to it in the dataset
		d([batch_size, 1]) - d_i = 1 if the ith instance is from labelled dataset
		r([batch_size, num_rules]) - the rule association matrix where r_ij = 1 if jth rule is associated with ith instance (r_ij = 1 => m_ij = 1)
		
		Output:
		loss(real number > 0) - the value of the LL_phi term
		'''
		psi = 1e-25
		L = tf.expand_dims(L,1)
		# [batch_size, num_rules]
		L = tf.tile(L,[1,self.num_rules])
		loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(tf.equal(l,L)),
													   logits=w_logits)
		loss = m*loss
		loss = (tf.to_float(tf.not_equal(l,L)) * loss) + (tf.to_float(tf.equal(l,L)) * r * loss)
# 		gcross_loss = gcross_utils.generalized_cross_entropy_bernoulli(weights,0.2)
		gcross_loss = generalized_cross_entropy_bernoulli(weights,0.2)
		gcross_loss = gcross_loss * m * tf.to_float(tf.equal(l,L)) * (1-r)
		loss = loss + gcross_loss
		loss = tf.reduce_sum(loss,axis=-1)
		loss = loss * d
		loss = tf.reduce_mean(loss)
		return loss


	def implication_loss(self, weights, f_probs, m, rule_classes, num_classes, d):
		''' 
		Func desc:
		Computes the implication loss value

		input:
		self object
		weights([batch_size, num_rules]) - the weights matrix corresponding to rule network(w_network) in the algorithm
		f_probs([batch_size, 1]) - the prob values from classification network (f_network)
		m([batch_size, num_rules]) - the rule coverage matrix where m_ij = 1 if jth rule covers ith instance
		rule_classes
		num_classes(non_negative integer) - number of available classes
		d([batch_size, 1]) - d_i = 1 if the ith instance is from labelled dataset

		output:
		-obj (real number) - the implication loss value
		'''
		# computes implication loss (Equation 4 in the paper)
		# weights are P_{j,\phi} values from the w network (rule network)
		# weights: [batch_size, num_rules] 
		# f_probs are probabilities from the f network (classification network)
		# f_probs: [batch_size, num_classes]
		psi = 1e-25 # a small value to avoid nans

		#[num_rules, num_classes]
		one_hot_mask = tf.one_hot(rule_classes,num_classes,dtype=tf.float32)
		#[batch_size, num_rules]
		f_probs = tf.matmul(f_probs, one_hot_mask, transpose_b=True)
		obj = 1 - (weights * (1 - f_probs)) #(Argument of log in equation 4)

		# computing last term of equation 5, will multiply with gamma outside this function
		obj = m*tf.log(obj + psi)
		obj = tf.reduce_sum(obj, axis=-1)
		obj = obj * (1-d) #defined only for instances in U, so mask by (1-d)
		obj = tf.reduce_mean(obj)
		return -obj

	def get_weights_and_logits_f_d_U(self, x):
		'''
		Func desc:
		compute and get the weights and logits for the rule network (w_network) 

		Input: 
		self object
		x([batch_size, num_features]) - instance matrix

		Output:
		weights([batch_size, num_rules]) - the r_ij values i.e. the possibility of a rule overfitting on an instance (r_ij = 0 for ith instance and jth rule)
		w_logits([batch_size, num_rules]) - 
		'''
		# Need to run the w network for each rule for the same x
		#
		# [batch_size, num_rules, num_features]
		x_shape = tf.shape(x)
		mul = tf.convert_to_tensor([1, self.num_rules_to_train])
		expanded_x = tf.tile(x, mul)
		# Need a python integer as the last dimension so that defining neural
		# networks work later. Hence use num_features instead of x_shape[1]
		x = tf.reshape(expanded_x , [x_shape[0], self.num_rules_to_train,
			self.num_features])

		batch_size = x_shape[0]
		rules_int = tf.convert_to_tensor([list(range(0,
			self.num_rules_to_train))])
		# Need to tile rules_int batch_size times
		#
		# tilevar should be a 1-D tensor with number of values equal to number
		# of columns in rules_int. Each column specifies the number of times
		# that axis in rules_int will be replicated.
		#
		# Following will replicate the rows of rules_int batch_size times and
		# leave the columns unchanged
		tilevar = tf.convert_to_tensor([batch_size, 1])
		rules_int = tf.tile(rules_int, tilevar) 
		rules_one_hot = tf.one_hot(rules_int, self.num_rules_to_train)
		rules_int = tf.expand_dims(rules_int, axis=-1)
		w_dict = {'x': x, 'rules' : rules_one_hot,
				'rules_int': rules_int}
		w_logits = self.w_network(w_dict, dropout_keep_prob=self.dropout_keep_prob)
		w_logits = tf.squeeze(w_logits)
		weights = tf.nn.sigmoid(w_logits)
		return weights, w_logits

	def joint_scores_from_f_and_w(self,weights,m,f_probs):
		'''
		Func desc:
		Compute the learning scores obtained while jointly learning f(classification network) and w(rule network)

		Input:
		self object
		weights([num_instances, num_rules]) - the weights matrix corresponding to rule network(w_network) in the algorithm
		m([batch_size, num_rules]) - the rule coverage matrix where m_ij denotes if jth rule covers ith instance (if yes, then m_ij = 1)
		f_probs([batch_size, 1]) - the prob values from classification network (f_network)

		Output:
		results([batch_size,1]) -  
		'''
		num_classes = self.num_classes
		rule_classes = self.rule_classes
		#[batch_size, num_rules, 1]
		weights = tf.expand_dims(weights,-1)
		weights_mask = tf.to_float(tf.greater(weights,0.5))
		#[batch_size, num_rules, 1]
		m = tf.expand_dims(m,-1)
		m = m*weights_mask
		#[num_rules, num_classes]
		one_hot_rule_classes = tf.one_hot(rule_classes,num_classes,dtype=tf.float32)
		#[1, num_rules, num_classes]
		one_hot_rule_classes = tf.expand_dims(one_hot_rule_classes,0)
		#[batch_size, num_rules, num_classes]
		rule_weight_product = weights * one_hot_rule_classes + (1-weights)*(1-one_hot_rule_classes)
		sum_rule_firings = tf.reduce_sum(m,1)
		result = m*rule_weight_product #+ (1-m)
		#[batch_size, num_classes]
		result = tf.reduce_sum(result,1)/(sum_rule_firings+1e-20)
		result = result + f_probs
		return result    
