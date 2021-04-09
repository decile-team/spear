from my_utils import get_data
import tensorflow as tf

class Implyloss:
	def __init__(self,data, num_classes):
		'''
		func desc:
		the constructor of the Implyloss class

		Input:
		data(9-length list): a list of the required values extracted from the pickle file
		num_classes(integer): the number of classes from which we can label

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
		'''
		assert (self.f_d_U_x[1]==num_features) # batch_size
		self.f_d_U_x=data[0]
		assert (self.f_d_U_l[0]==self.f_d_U_x[0]) # batch_size
		self.f_d_U_l=data[1]
		assert (m.shape==l.shape)
		self.f_d_U_m=data[2]
		assert (L.shape[0]==m.shape[0])
		self.f_d_U_L=data[3]
		assert(d.shape==L.shape)
		self.f_d_U_d=data[4]
		assert (r.shape==l.shape)
		self.f_d_U_r=data[5]
		self.f_d_U_s=data[6]
		self.f_d_U_n=data[7]
		self.f_d_U_k=data[8]
		self.f_d_U_num_features=self.f_d_U_x[1] # 1st dimension of x
		self.f_d_U_num_rules_to_train=self.f_d_U_l[1]
		self.f_d_U_num_classes = num_classes
		self.f_d_U_w_network = functools.partial(w_network, self.w_var_scope, self.f_d_U_num_rules)
		self.f_d_U_f_network = functools.partial(f_network, self.f_var_scope)
	
	def optimize(self):
		'''
		func desc: 
		compute the training objective and aim to minimize the loss function using the adam optimizer
		
		Input:
		self object

		evaluates:
		weights, w_logits of rule network(Used to train P_j_phi(r_j/x_i) i.e. whether rij = 1 for the ith instance and jth rule) [batch_size, num_rules]
		f_logits of the classification network (Used to train P_j_theta(l_j/x_i) i.e. the probability of ith instance belonging to jth class)
		LL_phi term
		LL_theta term
		training objective term
		minimize the loss using adam optimizer
		'''
		# w_network - rule network: Used to train P_j_phi(r_j/x_i) i.e. whether rij = 1 for the ith instance and jth rule
		# weights: [batch_size, num_rules]
		# w_logits: [batch_size, num_rules]
		weights, w_logits = self.get_weights_and_logits(self.f_d_U_x)
		self.f_d_U_weights = weights

		# f_network - classification network: Used to train P_j_theta(l_j/x_i) i.e. the probability of ith instance belonging to jth class 
		# f_dict: [batch_size, num_classes]
		f_dict = {'x': self.f_d_U_x}
		f_logits = self.f_network(f_dict, self.f_d_U_num_classes, reuse=True, dropout_keep_prob=self.f_d_U_dropout_keep_prob)
		self.f_d_U_probs = tf.math.softmax(f_logits, axis=-1) # value computed along axis = -1 => this dimension reduced in output
		self.f_d_U_preds = tf.argmax(self.f_d_U_probs, axis=-1) # was f_probs earlier, made probs now ?
		self.f_d_U_joint_f_w_score = self.joint_scores_from_f_and_w(self.f_d_U_weights,self.f_d_U_m,self.f_d_U_probs)

		# Do this so that the cross-entropy does not blow for data from U
		# The actual value of cross-entropy for U does not matter since it
		# will be multiplied by 0 anyway.
		L = L % self.f_d_U_num_classes

		# LL(\theta) (first term in eqn 5)
		# LL_theta term which is on d data
		L_one_hot = tf.one_hot(L, self.f_d_U_num_classes)
		LL_theta = tf.nn.softmax_cross_entropy_with_logits(logits=f_logits,
				labels=L_one_hot)
		LL_theta = d * LL_theta
		LL_theta = tf.reduce_mean(LL_theta) # loss of f network on labeled data d
		# loss of f network on labeled data d
		

		# LL(\phi) term (second term in eqn 5)
		LL_phi = self.compute_LL_phi(w_logits=w_logits, weights=self.f_d_U_weights, l=self.f_d_U_l, m=self.f_d_U_m, L=self.f_d_U_L, d=self.f_d_U_d, r=self.f_d_U_r)
		
		self.f_d_U_adam_lr = tf.placeholder(tf.float32,name='adam_lr')
		f_cross_training_optimizer = tf.train.AdamOptimizer(learning_rate=self.f_d_U_adam_lr, name='adam')

		training_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

		if 'implication' == self.config.mode:
			implication_loss = self.implication_loss(weights=self.f_d_U_weights,
													f_probs=self.f_d_U_probs,
													m=self.f_d_U_m,
													rule_classes=self.f_d_U_rule_classes,
													num_classes=self.f_d_U_num_classes,
													d=d)
			
			# (eqn 5)
			self.f_d_U_implication_loss = LL_phi \
										+ LL_theta \
										+ self.config.gamma*implication_loss

			with tf.control_dependencies([inc_f_d_U_global_step,  ]): # need to define inc_f_d_U_global_step here using f_d_U_train_ops and f_d_train_ops  
				self.f_d_U_implication_op = f_cross_training_optimizer.minimize( # made f_d_U_ in implication_op also ..check if required ?
						self.f_d_U_implication_loss,
						var_list=training_var_list) 
	
	# softmax_cross_entropy_with_logits,

	# get_weights_and_logits: Input - x [batch_size, num_rules], Output - weights [batch_size, num_rules], w_logits [batch_size, num_rules]
	def get_weights_and_logits(self, x):
		'''
		func desc:
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
		mul = tf.convert_to_tensor([1, self.f_d_U_num_rules_to_train])
		expanded_x = tf.tile(x, mul)
		# Need a python integer as the last dimension so that defining neural
		# networks work later. Hence use num_features instead of x_shape[1]
		x = tf.reshape(expanded_x , [x_shape[0], self.f_d_U_num_rules_to_train,
			self.f_d_U_num_features])

		batch_size = x_shape[0]
		rules_int = tf.convert_to_tensor([list(range(0,
			self.f_d_U_num_rules_to_train))])
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
		rules_one_hot = tf.one_hot(rules_int, self.f_d_U_num_rules_to_train)
		rules_int = tf.expand_dims(rules_int, axis=-1)
		w_dict = {'x': x, 'rules' : rules_one_hot,
				'rules_int': rules_int}
		w_logits = self.w_network(w_dict, dropout_keep_prob=self.f_d_U_dropout_keep_prob)
		w_logits = tf.squeeze(w_logits)
		weights = tf.nn.sigmoid(w_logits)
		return weights, w_logits

	# joint_scores_from_f_and_w: Input - weights [batch_size, num_rules], m [batch_size, num_rules], f_probs [batch_size, num_classes], result - scalar 
	def joint_scores_from_f_and_w(self,weights,m,f_probs):
		'''
		func desc:
		Compute the learning scores obtained while jointly learning f(classification network) and w(rule network)

		Input:
		self object
		weights([num_instances, num_rules]) - the weights matrix corresponding to rule network(w_network) in the algorithm
		m([batch_size, num_rules]) - the rule coverage matrix where m_ij denotes if jth rule covers ith instance (if yes, then m_ij = 1)
		f_probs([batch_size, 1]) - the prob values from classification network (f_network)

		Output:
		results([batch_size,1]) -  
		'''
		num_classes = self.f_d_U_num_classes
		rule_classes = self.f_d_U_rule_classes
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


	# compute_LL_phi
	def compute_LL_phi(self, w_logits, weights, l, m, L, d, r):
		'''
		func desc: 
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
		L = tf.tile(L,[1,self.f_d_U_num_rules])
		loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(tf.equal(l,L)),
													   logits=w_logits)
		loss = m*loss
		loss = (tf.to_float(tf.not_equal(l,L)) * loss) + (tf.to_float(tf.equal(l,L)) * r * loss)
		gcross_loss = gcross_utils.generalized_cross_entropy_bernoulli(weights,0.2)
		gcross_loss = gcross_loss * m * tf.to_float(tf.equal(l,L)) * (1-r)
		loss = loss + gcross_loss
		loss = tf.reduce_sum(loss,axis=-1)
		loss = loss * d
		loss = tf.reduce_mean(loss)
		assert (loss>=0)
		return loss
	
		# need to write loss functions

	def implication_loss(self, weights, f_probs, m, rule_classes, num_classes, d):
		''' 
		func desc:
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

