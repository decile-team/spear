import sys
sys.path.append('../../')

import re
import numpy as np

from spear.labeling import PreLabels


from lfs import rules, ClassLabels
from utils import load_data_to_numpy

X, X_feats, Y = load_data_to_numpy()
Y = np.array([ClassLabels[x].value for x in Y])

trec_noisy_labels = PreLabels(name="sms",
                               data=X,
                               gold_labels=Y,
                               data_feats=X_feats,
                               rules=rules,
                               labels_enum=ClassLabels,
                               num_classes=6)
L,S = trec_noisy_labels.get_labels()
# analyse = trec_noisy_labels.analyse_lfs(plot=True)