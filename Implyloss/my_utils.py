import pickle, os, json
import numpy as np
num_args = 9
def get_data(path):
	'''
	func desc:
	takes the pickle file and arranges it in a matrix list form so as to set the member variables accordingly
	expected order in pickle file is NUMPY arrays x, l, m, L, d, r, s, n, k
	x: [num_instances, num_features]
	l: [num_instances, num_rules]
	m: [num_instances, num_rules]
	L: [num_instances, 1]
	d: [num_instances, 1]
	r: [num_instances, num_rules]
	s: [num_instances, num_rules]
	n: [num_rules] Mask for s
	k: [num_rules] LF classes, range 0 to num_classes-1
	'''
	data=[]
	with open(path,'rb') as file:
		a=pickle.load(file)
		data.append(a) # check if this is required

	assert len(data)==num_args
	return data

def analyze_w_predictions(x,l,m,L,d,weights,probs,rule_classes):
	''' 
	func desc: 
	analyze the rule network by computing the precisions of the rules and comparing old and new rule stats

	input: 
	x: [num_instances, num_features]
	l: [num_instances, num_rules]
	m: [num_instances, num_rules]
	L: [num_instances, 1]
	d: [num_instances, 1]
	weights: [num_instances, num_rules]
	probs: [num_instances, num_classes]
	rule_classes: [num_rules,1]

	output:
	void, prints the required statistics
	'''
	num_classes = probs.shape[1]
	new_m = convert_weights_to_m(weights) * m
	new_l = convert_m_to_l(new_m,rule_classes,num_classes)
	o_micro,o_marco_p,o_rp = get_rule_precision(l,L,m)
	n_mirco,new_macro_p,n_rp = get_rule_precision(new_l,L,new_m)
	print("old micro precision: ", o_micro)
	print("new micro precision: ", n_mirco)
	print("old rule firings: ", np.sum(m))
	print("new rule firings: ", np.sum(new_m))
	print("old rule coverage: ", len([i for i in m if sum(i) > 0]))
	print("new rule coverage: ", len([i for i in new_m if sum(i) > 0]))

def convert_weights_to_m(weights):
	'''
	func desc:
	converts weights to m 
	
	input:
	weights([batch_size, num_rules]) - the weights matrix corresponding to rule network(w_network) in the algorithm

	output:
	m([batch_size, num_rules]) - the rule coverage matrix where m_ij = 1 if jth rule covers ith instance
	'''
	new_m = weights > 0.5
	new_m = new_m.astype(np.int32)
	return new_m

def convert_m_to_l(m,rule_classes,num_classes):
	'''
	func desc:
	converts m to l

	input:
	m([batch_size, num_rules]) - the rule coverage matrix where m_ij = 1 if jth rule covers ith instance
	rule_classes - 
	num_classes(non_negative integer) - number of available classes

	output:
	l([batch_size, num_rules]) - labels assigned by the rules
	'''
	rule_classes = np.array([rule_classes]*m.shape[0])
	l = m * rule_classes + (1-m)*num_classes
	return l

def get_rule_precision(l,L,m):
	'''
	func desc:
	get the precision of the rules

	input:
	l([batch_size, num_rules]) - labels assigned by the rules
	L([batch_size, 1]) - L_i = 1 if the ith instance has already a label assigned to it in the dataset
	m([batch_size, num_rules]) - the rule coverage matrix where m_ij = 1 if jth rule covers ith instance

	output:
	micro_p - 
	macro_p -
	comp - 
	'''
	L = L.reshape([L.shape[0],1])
	comp = np.equal(l,L).astype(np.float)
	comp = comp * m
	comp = np.sum(comp,0)
	support = np.sum(m,0)
	micro_p = np.sum(comp)/np.sum(support)
	macro_p = comp/(support + 1e-25)
	supported_rules = [idx for idx,support_val in enumerate(support) if support_val>0]
	macro_p = macro_p[supported_rules]
	macro_p = np.mean(macro_p)
	return micro_p,macro_p,comp/(support + 1e-25)


# from utils
def merge_dict_a_into_b(a, b):
	'''
	func desc:
	set the dict values of b to that of a

	input:
	a, b : dicts

	output:
	void
	'''
	for key in a:
		assert key not in b
		b[key] = a[key]

def print_tf_global_variables():
	'''
	Func Desc:
	prints all the global variables

	Input:

	Output:

	'''
	# import tensorflow as tf
	import tensorflow.compat.v1 as tf
	tf.disable_v2_behavior()
	print(json.dumps([str(foo) for foo in tf.global_variables()], indent=4))

def print_var_list(var_list):
	'''
	Func Desc:
	Prints the given variable list

	Input:
	var_list

	Output:

	'''
	print(json.dumps([str(foo) for foo in var_list], indent=4))

def pretty_print(data_structure):
	'''
	Func Desc:
	prints the given data structure in the desired format

	Input:
	data_structure

	Output:

	'''
	print(json.dumps(data_structure, indent=4))

def get_list_or_None(s, dtype=int):
	'''
	Func Desc:
	Returns the list of types of the variables in the string s

	Input:
	s - string
	dtype function (default - int)

	Output:
	None or list
	'''
	if s.strip() == '':
		return None
	else:
		lst = s.strip().split(',')
		return [dtype(x) for x in lst]

def get_list(s):
	'''
	Func Desc:
	returns the output of get_list_or_None as a list

	Input:
	s - list

	Output:
	lst - list
	'''
	lst = get_list_or_None(s)
	if lst is None:
		return []
	else:
		return lst

def None_if_zero(n):
	'''
	Func Desc:
	the max(0,n) function with none id n<=0

	Input:
	n - integer

	Output:
	if n>0 then n else None
	'''
	if n <= 0:
		return None
	else:
		return n

def boolean(s):
	'''
	Func Desc:
	returns the expected boolean value for the given string

	Input:
	s - string

	Output:
	boolean or error
	'''
	if s == 'True':
		return True
	if s == 'False':
		return False
	raise ValueError('Invalid boolean value: %s' % s)

def set_to_list_of_values_if_None_or_empty(lst, val, num_vals):
	'''
	Func Desc:
	returns lst if it is not empty else returns a same length list but with all its entries equal to val
	lst - list
	val - value 
	num_vals (integer) - length of the list lst

	Output:
	lst or same length val list
	'''
	if not lst:
		return [val] * num_vals
	else:
		print(len(lst), num_vals)
		assert len(lst) == num_vals
		return lst


# from snorkel_utils
def conv_l_to_lsnork(l,m):
	'''
	func desc:
	in snorkel convention
	if a rule does not cover an instance assign it label -1
	we follow the convention where we assign the label num_classes instead of -1
	valid class labels range from {0,1,...num_classes-1}
	conv_l_to_lsnork:  converts l in our format to snorkel's format

	input:
	l([batch_size, num_rules]) - rule label matrix
	m([batch_size, num_rules]) - rule coverage matrix
	
	output:
	lsnork([batch_size, num_rules])
	'''
	lsnork = l*m + -1*(1-m)
	return lsnork.astype(np.int)

# from metric_utils
def compute_accuracy(support, recall):
	'''
	func desc:
	compute the required accuracy 

	input:
	support 
	recall 

	output:
	accuracy
	'''
	return np.sum(support * recall) / np.sum(support)

# from data_utils
def dump_labels_to_file(save_filename, x, l, m, L, d, weights=None, f_d_U_probs=None, rule_classes=None):
	'''
	Func Desc:
	dumps the given data into a pickle file

	Input:
	save_filename - the name of the pickle file in which the arguments/data is required to be saved
	x ([batch_size x num_features])
	l ([batch_size x num_rules])
	m ([batch_size x num_rules])
	L ([batch_size x 1])
	d ([batch_size x 1])
	weights (default - None)
	f_d_U_probs (default - None)
	rule_classes  (default - None)

	Output:

	'''
	save_file = open(save_filename, 'wb')
	pickle.dump(x, save_file)
	pickle.dump(l, save_file)
	pickle.dump(m, save_file)
	pickle.dump(L, save_file)
	pickle.dump(d, save_file)

	if not weights is None:
		pickle.dump(weights, save_file)

	if not f_d_U_probs is None:
		pickle.dump(f_d_U_probs, save_file)

	if not rule_classes is None:
		pickle.dump(rule_classes,save_file)

	save_file.close()

def load_from_pickle_with_per_class_sampling_factor(fname, per_class_sampling_factor):
	'''
	Func Desc:
	load the data from the given pickle file with per class sampling factor

	Input:
	fname - name of the pickle file from which data need to be loaded
	per_class_sampling_factor 

	Output:
	the required matrices
	x1 ([batch_size x num_features])
	l1 ([batch_size x num_rules])
	m1 ([batch_size x num_rules])
	L1 ([batch_size x 1])
	d1 ([batch_size x 1])
	'''
	with open(fname, 'rb') as f:
		x = pickle.load(f)
		l = pickle.load(f)
		m = pickle.load(f)
		L = pickle.load(f)
		d = np.squeeze(pickle.load(f))

	x1 = []
	l1 = []
	m1 = []
	L1 = []
	d1 = []
	for xx, ll, mm, LL, dd in zip(x, l, m, L, d):
		for i in range(per_class_sampling_factor[LL]):
			x1.append(xx)
			l1.append(ll)
			m1.append(mm)
			L1.append(LL)
			d1.append(dd)

	x1 = np.array(x1)
	l1 = np.array(l1)
	m1 = np.array(m1)
	L1 = np.array(L1)
	d1 = np.array(d1)

	return x1, l1, m1, L1, d1


def combine_d_covered_U_pickles(d_name, infer_U_name, out_name, d_sampling_factor, U_sampling_factor):
	'''
	Func Desc:
	combine the labelled and unlabelled data, merge the corresponding parameters together and store them in new file

	Input:
	d_name - the pickle file storing labelled data
	infer_U_name - the pickle file storing unlabelled data
	out_name - the name of the file where merged output needs to be stored
	d_sampling_factor - the per_class_sampling_factor for labelled data
	U_sampling_factor - the per_class_sampling_factor for unlabelled data

	Output:

	'''
	#d_sampling_factor = np.array(d_sampling_factor)
	#U_sampling_factor = np.array(U_sampling_factor)

	d_x, d_l, d_m, d_L, d_d = load_from_pickle_with_per_class_sampling_factor(d_name, d_sampling_factor)
	U_x, U_l, U_m, U_L, U_d = load_from_pickle_with_per_class_sampling_factor(infer_U_name, U_sampling_factor)

	x = np.concatenate((d_x, U_x))
	l = np.concatenate((d_l, U_l))
	m = np.concatenate((d_m, U_m))
	L = np.concatenate((d_L, U_L))
	#print(d_d.shape)
	#print(U_d.shape)
	d = np.concatenate((d_d, U_d))

	with open(out_name, 'wb') as out_file:
		pickle.dump(x, out_file)
		pickle.dump(l, out_file)
		pickle.dump(m, out_file)
		pickle.dump(L, out_file)
		pickle.dump(d, out_file)

# from learn2reweight_utils
def updated_theta_copy(grads, variables, lr, mode):
	'''
	Func Desc:
	updates the theta (parameters) using rhe given learning rate, grads and variables

	Input:
	grads - gradients
	variables
	lr - learning rate
	mode 

	Output:
	vals - list of the updated gradients 
	'''
	vals = []
	if mode == 1:
		for g,v in zip(grads,variables):
			vals.append(v+lr*g)
	elif mode == -1:
		for g,v in zip(grads,variables):
			vals.append(v-lr*g)
	else:
		print("invalid mode error!")
		print(exit(1))

	return vals

