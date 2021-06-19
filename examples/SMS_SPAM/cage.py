import sys
sys.path.append('../../')
import numpy as np

from spear.cage import Cage
from spear.utils import get_data, get_classes

path_json = 'data_pipeline/sms_json.json'
U_path_pkl = 'data_pipeline/sms_pickle_U.pkl' #unlabelled data
T_path_pkl = 'data_pipeline/sms_pickle_T.pkl' #test data
log_path_1 = 'log/cage_log_1.txt'
n_lfs = 16

#---------------use of get_data, get_classes------------------
data = get_data(T_path_pkl, check_shapes=True) #check_shapes being True asserts, for relative shapes of arrays in pkl file
classes = get_classes(path_json)
print("Number of elements in data list: ", len(data))
print("Classes dictionary in json file(modified to have integer keys): ", classes)




cage = Cage(path_json, n_lfs)

#-----------FIT n PREDICT---------

#---------------use of fit_and_predict_proba------------------
probs = cage.fit_and_predict_proba(U_path_pkl, T_path_pkl, log_path_1)
labels = np.argmax(probs, 1)
print("probs shape: ", probs.shape)
print("labels shape: ",labels.shape)

#---------------use of fit_and_predict------------------
labels = cage.fit_and_predict(U_path_pkl, T_path_pkl, log_path_1, need_strings=False)
print("labels shape: ", labels.shape)

#---------------use of fit_and_predict------------------
labels_strings = cage.fit_and_predict(U_path_pkl, T_path_pkl, log_path_1, need_strings=True)
print("labels_strings shape: ", labels_strings.shape)


cage.save_params('params/sms_cage_params.pkl')
cage.load_params('params/sms_cage_params.pkl')

#-----------PREDICT---------

#---------------use of predict_proba------------------
probs_test = cage.predict_proba(T_path_pkl)
print("probs_test shape: ",probs_test.shape)

#---------------use of predict------------------
labels_test = cage.predict(T_path_pkl, need_strings = False)
print("labels_test shape: ", labels_test.shape)