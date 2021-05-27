import pickle
import numpy as np 
import torch
from torch.distributions.beta import Beta
import tensorflow as tf

def get_data(path):
	'''
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
	data = []
	with open(path, 'rb') as file:
		contents = pickle.load(file)
		for i in range(9):
			if i == 0:
				data.append(pickle.load(f))
			else if i == 6:
				data.append(pickle.load(f).astype(np.float32))
			else:
				data.append(pickle.load(f).astype(np.int32))
	return data

	import tensorflow as tf

def updated_theta_copy(grads, variables, lr, mode):
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
