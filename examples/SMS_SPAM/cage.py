import sys
sys.path.append('../../')

from spear.Cage import Cage

path_json = 'data_pipeline/sms_json.json'
U_path_pkl = 'data_pipeline/sms_pickle_U.pkl'
T_path_pkl = 'data_pipeline/sms_pickle_T.pkl'
log_path_1 = 'log/cage_log_1.txt'

cage = Cage(path_json, U_path_pkl)
probs = cage.fit_and_predict_proba(T_path_pkl, log_path_1)
