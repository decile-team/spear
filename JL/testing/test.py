import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from core import *
from subset_selection import *

if __name__ == '__main__':
	n_classes = 2
	n_lfs = 10
	file_L = 'pickled_data/data_L.pkl'
	file_U = 'pickled_data/data_U.pkl'
	file_V = 'pickled_data/data_V.pkl'
	file_T = 'pickled_data/data_T.pkl'
	mask = [1,1,1,1,1,1,1]
	batch_size = 32
	lr_feature = 0.0003
	lr_gm = 0.01
	path_log = 'logs/yt_unsup.txt'

	jl = JL(n_classes, file_L, file_U, file_V, file_T, True)
	fm, gm = jl.fit(mask, batch_size, lr_feature, lr_gm, path_log, True, 100, -1, 7, True, True, 0.9, 0.85, 0, 'lr', 'macro')
	print(fm.shape)
	print(gm.shape)