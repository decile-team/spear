import sys
sys.path.append('../../')

import numpy as np

from spear.labeling import PreLabels


from lfs import rules, ClassLabels
from utils import load_data_to_numpy, get_various_data

X, X_feats, Y = load_data_to_numpy()

validation_size = 100
test_size = 200
L_size = 100
U_size = X.size - L_size - validation_size - test_size
# U_size = 300

X_V,Y_V,X_feats_V,R_V, X_T,Y_T,X_feats_T,R_T, X_L,Y_L,X_feats_L,R_L, X_U,X_feats_U,R_U = get_various_data(X,Y,\
    X_feats, len(rules.get_lfs()),validation_size,test_size,L_size,U_size)


sms_noisy_labels = PreLabels(name="sms",
                               data=X_V,
                               gold_labels=Y_V,
                               data_feats=X_feats_V,
                               rules=rules,
                               labels_enum=ClassLabels,
                               num_classes=2)
# L,S = sms_noisy_labels.get_labels()
# analyse = sms_noisy_labels.analyse_lfs(plot=True)
sms_noisy_labels.generate_pickle('data_pipeline/sms_pickle_V.pkl')
sms_noisy_labels.generate_json('data_pipeline/sms_json.json') #JSON

sms_noisy_labels = PreLabels(name="sms",
                               data=X_T,
                               gold_labels=Y_T,
                               data_feats=X_feats_T,
                               rules=rules,
                               labels_enum=ClassLabels,
                               num_classes=2)
# L,S = sms_noisy_labels.get_labels()
# analyse = sms_noisy_labels.analyse_lfs(plot=True)
sms_noisy_labels.generate_pickle('data_pipeline/sms_pickle_T.pkl')

sms_noisy_labels = PreLabels(name="sms",
                               data=X_L,
                               gold_labels=Y_L,
                               data_feats=X_feats_L,
                               rules=rules,
                               labels_enum=ClassLabels,
                               num_classes=2)
# L,S = sms_noisy_labels.get_labels()
# analyse = sms_noisy_labels.analyse_lfs(plot=True)
sms_noisy_labels.generate_pickle('data_pipeline/sms_pickle_L.pkl')

sms_noisy_labels = PreLabels(name="sms",
                               data=X_U,
                               rules=rules,
                               data_feats=X_feats_U,
                               labels_enum=ClassLabels,
                               num_classes=2)
# L,S = sms_noisy_labels.get_labels()
# analyse = sms_noisy_labels.analyse_lfs(plot=True)
sms_noisy_labels.generate_pickle('data_pipeline/sms_pickle_U.pkl')
