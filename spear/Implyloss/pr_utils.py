import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

def exp_term_for_constraints(rule_classes, num_classes, C):
	'''
	Func Desc:
	Compute the exponential term for the constraints

	Input:
	rule_classes ([num_rules,1]) - a list of classes associated with the rules
	num_classes (int)
	C

	Output:
	the required exponential term
	'''
	rule_classes_tensor = tf.to_float(tf.convert_to_tensor(rule_classes))
	#rule_classes_tensor = tf.reshape(rule_classes_tensor,[1,rule_classes])
	rule_classes_tensor = tf.expand_dims(rule_classes_tensor,0)
	class_types_tensor = tf.to_float(tf.convert_to_tensor(np.arange(num_classes).reshape(num_classes,1)))	
	#[num_classes,num_rules]
	class_rule_constraint = tf.to_float(tf.equal(class_types_tensor,rule_classes_tensor)) - 1.0
	class_rule_constraint = tf.exp(C*class_rule_constraint)
	return class_rule_constraint


def pr_product_term(weights, rule_classes, num_classes, C):
	'''
	Func Desc:
	Compute the probability product term for the constraints

	Input:
	weights ([batch_size, num_rules]) - the w_network weights
	rule_classes ([num_rules,1]) - a list of classes associated with the rules
	num_classes (int)
	C

	Output:
	the required product term
	'''
	# weights: [batch_size, num_rules]
	class_rule_constraint = exp_term_for_constraints(rule_classes, num_classes, C)
	#class_rule_constraint = tf.Print(class_rule_constraint,[tf.shape(class_rule_constraint)],message="class_rule_constraint")
	#[num_classes,1,num_rules]
	class_rule_constraint = tf.expand_dims(class_rule_constraint,axis=1)
	#[1,batch_size,num_rules]
	weights = tf.expand_dims(weights,axis=0) 	 
	# [num_classes,batch_size,num_rules]
	t1 = class_rule_constraint * weights
	# [1, batch_size, num_rules]
	t2 = 1-weights
	#[num_classes,batch_size,num_rules]
	t = t1+t2
	#t = tf.Print(t, [t,tf.shape(t)],message="t and shape of t")
	product_term = tf.reduce_prod(t,axis=-1)
	#[batch_size, num_classes]
	product_term = tf.transpose(product_term)
	return product_term


def get_q_y_from_p(f_probs, weights, rule_classes, num_classes, C):
	'''
	Func Desc:
	Compute the q_y term from the p (f_network) distribution 

	Input:
	f_probs ([batch_size, num_classes])
	weights ([batch_size, num_rules]) - the w_network weights
	rule_classes ([num_rules,1]) - a list of classes associated with the rules
	num_classes (int)
	C

	Output:
	the required q_y term
	'''
	# f_probs: [batch_size, num_classes]
	# weights: [batch_size, num_rules]
	psi = 1e-20
	product_term = pr_product_term(weights, rule_classes, num_classes, C)
	result = f_probs * product_term
	normalizer = tf.reduce_sum(result,axis=-1,keepdims=True)
	result = result/(normalizer + psi)
	return result

def get_q_r_from_p(f_probs, weights, rule_classes, num_classes, C):
	'''
	Func Desc:
	Compute the q_r term from the p (f_network) distribution 

	Input:
	f_probs ([batch_size, num_classes])
	weights ([batch_size, num_rules]) - the w_network weights
	rule_classes ([num_rules,1]) - a list of classes associated with the rules
	num_classes (int)
	C

	Output:
	the required q_r term
	'''
	# f_probs: [batch_size, num_classes]
	# weights: [batch_size, num_rules]
	psi = 1e-20
	#[batch_size, num_classes]
	pr_product_t = pr_product_term(weights, rule_classes, num_classes, C)
	#[batch_size, 1, num_classes] 
	product_term = tf.expand_dims(pr_product_t,axis=1)
	#[num_rules, num_classes]
	class_rule_constraint = tf.transpose(exp_term_for_constraints(rule_classes, num_classes, C))
	#[1, num_rules, num_classes]
	class_rule_constraint = tf.expand_dims(class_rule_constraint, axis=0)
	#[batch_size, num_rules, 1]
	w = tf.expand_dims(weights,2)
	#[batch_size, num_rules, num_classes]
	divisior = w*class_rule_constraint + (1-w)
	product_term = product_term / (divisior + psi)
	#[batch_size, 1, num_classes]
	f_probs = tf.expand_dims(f_probs,axis=1)
	#[batch_size, num_rules, num_classes]
	product_term = product_term * f_probs * class_rule_constraint
	#[batch_size, num_rules]
	sum_over_y_term = tf.reduce_sum(product_term,axis=-1)
	prob_q_r_eq_1 = weights * sum_over_y_term


	prob_q_r_eq_0 = tf.reduce_sum(f_probs * product_term, axis=-1)
	prob_q_r_eq_0 = (1 - weights) * prob_q_r_eq_0

	prob_q_r_eq_1 = prob_q_r_eq_1 / (prob_q_r_eq_0 + prob_q_r_eq_1)

	return prob_q_r_eq_1


def theta_term_in_pr_loss(f_logits, f_probs, weights, rule_classes, num_classes, C, d):
	'''
	Func Desc:
	Compute the theta term in the pr loss 

	Input:
	f_logits ([batch_size, num_classes])
	f_probs ([batch_size, num_classes])
	weights ([batch_size, num_rules]) - the w_network weights
	rule_classes ([num_rules,1]) - a list of classes associated with the rules
	num_classes (int)
	C
	d ([batch_size,1])

	Output:
	the required theta term (third term in equation 14) -  used to supervise f (classification) network from instances in U
	'''
	#[batch_size, num_classes]
	q_y = get_q_y_from_p(f_probs, weights, rule_classes, num_classes, C)
	cross_ent_q_p = tf.nn.softmax_cross_entropy_with_logits(labels=q_y,logits=f_logits)
	cross_ent_q_p = (1-d) * cross_ent_q_p #defined only for instances in U, so mask by (1-d)
	result = tf.reduce_mean(cross_ent_q_p)
	return result

def phi_term_in_pr_loss(m, w_logits, f_probs, weights, rule_classes, num_classes, C, d):
	'''
	Func Desc:
	Compute the phi term in the pr loss

	Input:
	m ([batch_size, num_rules]) - mij = 1 if ith example is associated with jth rule 
	w_logits ([batch_size, num_rules])
	f_probs ([batch_size, num_classes])
	weights ([batch_size, num_rules]) - the w_network weights
	rule_classes ([num_rules,1]) - a list of classes associated with the rules
	num_classes (int)
	C
	d ([batch_size,1])

	Output:
	the required phi term (fourth term in equation 14) - used to superwise w (rule) network from instances in U
	'''
	#w_logits: [batch_size, num_rules]
	#m: [batch_size, num_rules]
	psi = 1e-20
	q_r_1 = get_q_r_from_p(f_probs, weights, rule_classes, num_classes, C)
	#[batch_size, num_rules]
	cross_ent_q_w = tf.nn.sigmoid_cross_entropy_with_logits(labels=q_r_1, logits=w_logits)
	cross_ent_q_w = tf.reduce_sum(cross_ent_q_w*m,axis=-1)
	#normalizer_cross_ent_q_w = tf.reduce_sum(m,axis=-1)
	#cross_ent_q_w = cross_ent_q_w / (normalizer_cross_ent_q_w + psi)
	cross_ent_q_w = cross_ent_q_w * (1-d)
	cross_ent_q_w = tf.reduce_mean(cross_ent_q_w)
	return cross_ent_q_w

def pr_loss(m, f_logits, w_logits, f_probs, weights, rule_classes, num_classes, C, d):
	'''
	Func Desc:
	Compute the  pr loss 

	Input:
	m ([batch_size, num_rules]) - mij = 1 if ith example is associated with jth rule 
	f_logits 
	w_logits ([batch_size, num_rules]) - logit before sigmoid activation in w network
	f_probs ([batch_size, num_classes]) - output of f network
	weights ([batch_size, num_rules]) - the sigmoid output from w network
	rule_classes ([num_rules,1]) - a list of classes associated with the rules
	num_classes (int)
	C - lamda in equation 10 (hyperparameter)
	d ([batch_size,1]) - if ith instance is from "d" set (labelled data) d[i] = 1, else if ith instance is from "U" set, d[i] = 0

	Output:
	the required phi term
	'''

	#theta_term : (third term in equation 14) (used to supervise f (classification) network from instances in U )
	#phi term : (fourth term in equation 14)  (used to superwise w (rule) network from instances in U )

	# m : rule_firing matrix: [batch_size, num_rules]
	# w_logits: logit before sigmoid activation in w network: [batch_size, num_rules]
	# weights: sigmoid output from w network: [batch_size, num_rules]
	# f_probs: output of f network: [batch_size, num_classes]
	# C: \lamda in equation 10 (hyperparameter)
	# d : [batch_size], d[i] = 0 if ith instance is from "U" set, 1 if ith instance is from "d" set (labeled data)

	theta_term = theta_term_in_pr_loss(f_logits, f_probs, weights, rule_classes, num_classes, C, d)
	cross_ent_q_w = phi_term_in_pr_loss(m, w_logits, f_probs, weights, rule_classes, num_classes, C, d)
	return theta_term + cross_ent_q_w




