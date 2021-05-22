from labeling.apply import *
from labeling.noisy_labels import *
from labeling.analysis import LFAnalysis
from lfs import rules, ClassLabels
from utils import load_data_to_numpy

import re

X, Y = load_data_to_numpy()
R = np.zeros((X.shape[0],len(rules.get_lfs())))

sms_noisy_labels = NoisyLabels(name="sms",
                               data=X,
                               gold_labels=Y,
                               rules=rules,
                               labels_enum=ClassLabels,
                               num_classes=2)

L,S = sms_noisy_labels.get_labels()
analyse = sms_noisy_labels.analyse_lfs(plot=True)
sms_noisy_labels.generate_pickle()
sms_noisy_labels.generate_json()