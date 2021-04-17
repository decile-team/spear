import numpy as np 
import pickle

if __name__ == "__main__":
	files = ["d_processed.p", "U_processed.p", "validation_processed.p", "test_processed.p"]
	output = ['pickled_data/data_L.pkl', 'pickled_data/data_U.pkl', 'pickled_data/data_V.pkl', 'pickled_data/data_T.pkl']
	files = ['Data/Youtube/'+file for file in files]
	#need to do subset selection here, but maybe done in core.py
	for num in range(len(files)):
		objs = []
		with open(files[num], 'rb') as f:
			while 1:
				try:
					o = pickle.load(f)
				except EOFError:
					break
				if num == 2:
					objs.append(o[:100])
				else:
					objs.append(o)
		objs[3] = objs[3].reshape((objs[3]).size, 1)
		objs[4] = objs[4].reshape((objs[4]).size, 1)
		print(len(objs))
		temp_file = open(output[num], 'wb')
		for obj in objs:
			pickle.dump(obj, temp_file)
		pickle.dump(objs[2], temp_file)
		pickle.dump(np.zeros(10), temp_file)
		pickle.dump(np.array([1, 1, 1, 1, 0, 1, 0, 0, 0, 0]), temp_file)
		temp_file.close()

