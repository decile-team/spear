import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from core import *

'''
Goal:
	to get the results from spear library into a log file and then check if they are
	 consistent with results from https://github.com/oishik75/CAGE

	Comment and uncomment the first two code snippets below for sms or spouse testing
'''


if __name__ == "__main__":

	#sms data
	# n_classes = 2
	# n_lfs = 11
	# train_file = "pickled_data/sms_train.pkl"
	# test_file = "pickled_data/sms_test.pkl"
	# log_file = "logs/sms.txt"

	#spouse data
	n_classes = 2
	n_lfs = 10
	train_file = "pickled_data/spouse_train.pkl"
	test_file = "pickled_data/spouse_test.pkl"
	log_file = "logs/spouse.txt"

	a = (np.ones(n_lfs)) * 0.9  # Quality  Guide all set to 0.9

	cage = Cage(n_classes, train_file, a, 0.85)
	cage.fit(test_file, log_file)