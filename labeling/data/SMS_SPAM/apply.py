from labeling.apply import *
from labeling.analysis import *

from lfs import rules, ClassLabels
from utils import load_data_to_numpy
import re

X, Y = load_data_to_numpy()
applier = LFApplier(lf_set = rules)

L,S = applier.apply(X)

analysis = LFAnalysis(ClassLabels,L,list(rules.get_lfs()))
df = analysis.lf_summary(plot=True)


# X, Y = load_data_to_numpy()
# R = np.zeros((X.shape[0],len(rules.get_lfs())))

# sms_noisy_labels = NoisyLabels("sms",X,Y,rules,R,ClassLabels)
# E,S = sms_noisy_labels.get_labels()
# sms_noisy_labels.generate_pickle()    