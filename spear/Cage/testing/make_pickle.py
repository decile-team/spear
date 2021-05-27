import numpy as np 
import pickle

'''
Goal:
	to make pickle file of the standard format from the data available in https://github.com/oishik75/CAGE

	Comment and uncomment the first two code snippets below for sms or spouse pickled data
'''

if __name__ == "__main__":


	#sms:
	n_classes = 2
	n_lfs = 11
	save_train_file = "pickled_data/sms_train.pkl"
	save_test_file = "pickled_data/sms_test.pkl"
	train_file = "Data/sms/train_L_S_smooth.npy"
	test_file = "Data/sms/test_L_S_smooth.npy"
	k_file = "Data/sms/k.npy"
	true_labels_file = "Data/sms/true_labels_sms.npy"
	continuous_mask = np.ones(n_lfs)  # All labeling functions are continuous

	#spouse:
	# n_classes = 2
	# n_lfs = 10
	# save_train_file = "pickled_data/spouse_train.pkl"
	# save_test_file = "pickled_data/spouse_test.pkl"
	# train_file = "Data/spouse/train_L_S_smooth.npy"
	# test_file = "Data/spouse/test_L_S_smooth.npy"
	# k_file = "Data/spouse/k.npy" #label for each LF
	# true_labels_file = "Data/spouse/true_labels_test.npy"
	# continuous_mask =np.array([0, 0, 1, 1, 0, 1, 1, 1, 1, 1])

#---------------------------------------------------------------------#

	n_features = 10

	# Discrete lambda values
	l = np.abs(np.load(train_file)[:, 0])
	n_instances, n_features = l.shape
	l_test = np.abs(np.load(test_file)[:, 0])
	n_instances_2 = l_test.shape[0]

	# Continuous score values
	s = np.load(train_file)[:, 1]
	s_test = np.load(test_file)[:, 1]

	# Labeling Function Classes
	k = np.load(k_file)

	# True y
	y_true_test = np.load(true_labels_file)

	train_file_ = open(save_train_file, "wb")
	pickle.dump(np.zeros((n_instances, n_features)), train_file_)
	pickle.dump(l, train_file_)
	pickle.dump(l, train_file_)
	pickle.dump(np.zeros((n_instances, 1)), train_file_)
	pickle.dump(np.zeros((n_instances, 1)), train_file_)
	pickle.dump(np.zeros((n_instances, n_lfs)), train_file_)
	pickle.dump(s, train_file_)
	pickle.dump(continuous_mask, train_file_)
	pickle.dump(k, train_file_)
	train_file_.close()

	test_file_ = open(save_test_file, "wb")
	pickle.dump(np.zeros((n_instances_2, n_features)), test_file_)
	pickle.dump(l_test, test_file_)
	pickle.dump(l_test, test_file_)
	pickle.dump(y_true_test.reshape(y_true_test.size,1), test_file_)
	pickle.dump(np.zeros((n_instances_2, 1)), test_file_)
	pickle.dump(np.zeros((n_instances_2, n_lfs)), test_file_)
	pickle.dump(s_test, test_file_)
	pickle.dump(continuous_mask, test_file_)
	pickle.dump(k, test_file_)
	test_file_.close()

