import sys
sys.path.append('../../')

from spear.JL import JL

path_json = 'data_pipeline/sms_json.json'
U_path_pkl = 'data_pipeline/sms_pickle_U.pkl'
L_path_pkl = 'data_pipeline/sms_pickle_L.pkl'
V_path_pkl = 'data_pipeline/sms_pickle_V.pkl'
T_path_pkl = 'data_pipeline/sms_pickle_T.pkl'

log_path_1 = 'log/jl_log_1.txt'

loss_func_mask = [1,1,1,1,1,1,0]
batch_size = 100
lr_fm = 0.0005
lr_gm = 0.01
use_accuracy_score = False
return_gm = True


jl = JL(path_json, L_path_pkl, U_path_pkl, V_path_pkl, T_path_pkl)
probs_fm, probs_gm = jl.fit_and_predict_proba(loss_func_mask, batch_size, lr_fm, lr_gm, use_accuracy_score, log_path_1, return_gm)