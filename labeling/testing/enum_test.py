from labeling.lf import *
from labeling.apply import *
from labeling.preprocess import *
from labeling.continuous_scoring import *
from labeling.noisy_labels import *
from labeling.lf_set import *
import numpy as np
import enum

class ClassLabels(enum.Enum):
    SPAM = 1
    HAM = 0

pre_resources={"r0":1.0}

@preprocessor(resources=pre_resources)
def square(x,**kwargs):
    return {"value":x*x*kwargs["r0"]}

cf_resources={"r1":4, "r2":8, "len1":4}
lf_resources={"r3":4, "len2":5}

@continuous_scorer(resources=cf_resources)
def score(x, **kwargs):
    t1=np.exp(-1*np.linalg.norm(x['value']))
    t2=(kwargs["r1"]+kwargs["r2"])/(kwargs["len1"]*kwargs["len1"])
    t3=kwargs["r3"]/kwargs["len2"]
    return t1*t2*t3

@labeling_function(pre=[square], resources=lf_resources, cont_scorer=score, label=ClassLabels.HAM.value)
def lf1(x, **kwargs):
    if np.linalg.norm(x['value']) < 1 and kwargs["r3"]==4:
        return ClassLabels.HAM
    return ABSTAIN

@labeling_function(pre=[square], label=1)                # no continuous scorer specified
def lf2(x, **kwargs):
    if np.linalg.norm(x['value']) < 5:
        return ClassLabels.SPAM
    return ABSTAIN

lfs = [lf1, lf2]
rules = LFSet("myrules")
rules.add_lf_list(lfs)

dataX = np.array([[0.48166037, 0.57330743, 0.06621459, 0.3704664],
                  [0.99777641, 0.87790733, 0.67211584, 0.46130919]])

dataY = np.array([0, 1])

applier = LFApplier(lf_set=rules)
E,S=applier.apply(dataX)

print(E)
print(S)

test_data_noisy_labels = NoisyLabels("testdata", dataX, dataY, rules)
E,S = test_data_noisy_labels.get_labels()

print(E)
print(S)

test_data_noisy_labels.generate_pickle("enumpkl")
f=open("enumpkl","rb")
noisy_data = pickle.load(f)
print(noisy_data["e"])