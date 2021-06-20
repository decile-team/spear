import sys
sys.path.append('../../')
import numpy as np

from spear.jl import JL

path_json = 'data_pipeline/sms_json.json'
U_path_pkl = 'data_pipeline/sms_pickle_U.pkl'
L_path_pkl = 'data_pipeline/sms_pickle_L.pkl'
V_path_pkl = 'data_pipeline/sms_pickle_V.pkl'
T_path_pkl = 'data_pipeline/sms_pickle_T.pkl'

log_path_1 = 'log/jl_log_1.txt'

loss_func_mask = [1,1,1,1,1,1,1]
batch_size = 100
lr_fm = 0.0005
lr_gm = 0.01
use_accuracy_score = False
n_lfs = 16
n_features = 1024

jl = JL(path_json, n_lfs, n_features)

#-----------FIT n PREDICT---------

#---------------use of fit_and_predict_proba------------------
return_gm = True
probs_fm, probs_gm = jl.fit_and_predict_proba(L_path_pkl, U_path_pkl, V_path_pkl, T_path_pkl, loss_func_mask, batch_size, lr_fm, lr_gm, use_accuracy_score, log_path_1, return_gm)
labels = np.argmax(probs_fm, 1)
print("probs_fm shape: ", probs_fm.shape)
print("probs_gm shape: ", probs_gm.shape)

#---------------use of fit_and_predict_proba------------------
return_gm = False
probs_fm = jl.fit_and_predict_proba(L_path_pkl, U_path_pkl, V_path_pkl, T_path_pkl, loss_func_mask, batch_size, lr_fm, lr_gm, use_accuracy_score, log_path_1, return_gm)
labels = np.argmax(probs_fm, 1)
print("probs_fm shape: ", probs_fm.shape)

#---------------use of fit_and_predict------------------
return_gm = True
probs_fm, probs_gm = jl.fit_and_predict(L_path_pkl, U_path_pkl, V_path_pkl, T_path_pkl, loss_func_mask, batch_size, lr_fm, lr_gm, use_accuracy_score, log_path_1, return_gm, need_strings = False)
print("probs_fm shape: ", probs_fm.shape)
print("probs_gm shape: ", probs_gm.shape)

#---------------use of fit_and_predict------------------
return_gm = True
probs_fm, probs_gm = jl.fit_and_predict(L_path_pkl, U_path_pkl, V_path_pkl, T_path_pkl, loss_func_mask, batch_size, lr_fm, lr_gm, use_accuracy_score, log_path_1, return_gm, need_strings = True)
print("probs_fm shape: ", probs_fm.shape)
print("probs_gm shape: ", probs_gm.shape)


jl.save_params('params/sms_jl_params.pkl')
jl.load_params('params/sms_jl_params.pkl')


#-----------PREDICT---------

#---------------use of predict_fm/gm_proba------------------
probs_fm_test = jl.predict_fm_proba(T_path_pkl)
probs_gm_test = jl.predict_gm_proba(T_path_pkl)
print("probs_fm_test shape: ", probs_fm_test.shape)
print("probs_gm_test shape: ", probs_gm_test.shape)

#---------------use of predict------------------
probs_fm_test = jl.predict_fm(T_path_pkl, need_strings=False)
probs_gm_test = jl.predict_gm(T_path_pkl, need_strings=False)
print("probs_fm_test shape: ", probs_fm_test.shape)
print("probs_gm_test shape: ", probs_gm_test.shape)