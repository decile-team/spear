from labeling.apply import *
from labeling.noisy_labels import *
from lfs import rules
from utils import load_data_to_numpy
import re

X, Y = load_data_to_numpy()
R = np.zeros((X.shape[0],len(rules.get_lfs())))

sms_noisy_labels = NoisyLabels("sms",X,Y,rules,R)
L,S = sms_noisy_labels.get_labels()
sms_noisy_labels.generate_pickle()

