from my_utils import get_data
from my_core import Implyloss

num_classes = 6
if __name__ == '__main__':
	path = "d_processed.p" # need to change this
	data = get_data(path) # path will be the path of pickle file
	Il = Implyloss(data,num_classes)
	Il.optimize()