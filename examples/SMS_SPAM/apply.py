import sys
sys.path.append('../../')

import re
import numpy as np

from spear.labeling import NoisyLabels


from lfs import rules, ClassLabels
from utils import load_data_to_numpy

X, X_feats, Y = load_data_to_numpy()

test_size = 200
validation_size = 100
L_size = 100
# U_size = X.size - L_size - validation_size - test_size
U_size = 300


index = np.arange(X.size)
index = np.random.permutation(index)
X = X[index]
Y = Y[index]
X_feats = X_feats[index]

X_V = X[-validation_size:]
Y_V = Y[-validation_size:]
X_feats_V = X_feats[-validation_size:]
R_V = np.zeros((validation_size,len(rules.get_lfs())))

X_T = X[-(validation_size+test_size):-validation_size]
Y_T = Y[-(validation_size+test_size):-validation_size]
X_feats_T = X_feats[-(validation_size+test_size):-validation_size]
R_T = np.zeros((test_size,len(rules.get_lfs())))

X_L = X[-(validation_size+test_size+L_size):-(validation_size+test_size)]
Y_L = Y[-(validation_size+test_size+L_size):-(validation_size+test_size)]
X_feats_L = X_feats[-(validation_size+test_size+L_size):-(validation_size+test_size)]
R_L = np.zeros((L_size,len(rules.get_lfs())))

# X_U = X[:-(validation_size+test_size+L_size)]
X_U = X[:U_size]
X_feats_U = X_feats[:U_size]
# Y_U = Y[:-(validation_size+test_size+L_size)]
R_U = np.zeros((U_size,len(rules.get_lfs())))


sms_noisy_labels = NoisyLabels(name="sms",
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

sms_noisy_labels = NoisyLabels(name="sms",
                               data=X_T,
                               gold_labels=Y_T,
                               data_feats=X_feats_T,
                               rules=rules,
                               labels_enum=ClassLabels,
                               num_classes=2)
# L,S = sms_noisy_labels.get_labels()
# analyse = sms_noisy_labels.analyse_lfs(plot=True)
sms_noisy_labels.generate_pickle('data_pipeline/sms_pickle_T.pkl')

sms_noisy_labels = NoisyLabels(name="sms",
                               data=X_L,
                               gold_labels=Y_L,
                               data_feats=X_feats_L,
                               rules=rules,
                               labels_enum=ClassLabels,
                               num_classes=2)
# L,S = sms_noisy_labels.get_labels()
# analyse = sms_noisy_labels.analyse_lfs(plot=True)
sms_noisy_labels.generate_pickle('data_pipeline/sms_pickle_L.pkl')

sms_noisy_labels = NoisyLabels(name="sms",
                               data=X_U,
                               rules=rules,
                               data_feats=X_feats_U,
                               labels_enum=ClassLabels,
                               num_classes=2)
# L,S = sms_noisy_labels.get_labels()
# analyse = sms_noisy_labels.analyse_lfs(plot=True)
sms_noisy_labels.generate_pickle('data_pipeline/sms_pickle_U.pkl')
