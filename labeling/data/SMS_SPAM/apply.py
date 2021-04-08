from labeling.apply import *
from labeling.analysis import *

from lfs import rules
from utils import load_data_to_numpy
import re

X, Y = load_data_to_numpy()
applier = LFApplier(lf_set = rules)

L,S = applier.apply(X)

analysis = LFAnalysis(L,list(rules.get_lfs()))
df = analysis.lf_summary(plot=True)


    