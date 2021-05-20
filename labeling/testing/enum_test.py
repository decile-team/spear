from labeling.lf import *
from labeling.apply import *
from labeling.preprocess import *
from labeling.continuous_scoring import *
from labeling.noisy_labels import *
from labeling.lf_set import *
import numpy as np
import enum, pickle, json

# define the ClassLabels enum
class ClassLabels(enum.Enum):
    SPAM = 1
    HAM = 0

# some useful resources
pre_resources={"r0":1.0}
cf_resources={"r1":4, "r2":8, "len1":4}
lf_resources={"r3":4, "len2":5}

# define preprocessor
@preprocessor(resources=pre_resources)
def square(x,**kwargs):
    return {"value":x*x*kwargs["r0"]}

# define continuous scorer
@continuous_scorer(resources=cf_resources)
def score(x, **kwargs):
    t1=np.exp(-1*np.linalg.norm(x['value']))
    t2=(kwargs["r1"]+kwargs["r2"])/(kwargs["len1"]*kwargs["len1"])
    t3=kwargs["r3"]/kwargs["len2"]
    return t1*t2*t3

# define labeling function
@labeling_function(pre=[square], resources=lf_resources, cont_scorer=score, label=ClassLabels.HAM)
def lf1(x, **kwargs):
    if np.linalg.norm(x['value']) < 1 and kwargs["r3"]==4:
        return ClassLabels.HAM
    return ABSTAIN

@labeling_function(pre=[square], label=ClassLabels.SPAM)                # no continuous scorer specified
def lf2(x, **kwargs):
    if np.linalg.norm(x['value']) < 5:
        return ClassLabels.SPAM
    return ABSTAIN

# create set of labeling functions to use for labeling the data
lfs = [lf1, lf2]
rules = LFSet("myrules")
rules.add_lf_list(lfs)

# the data
dataX = np.array([[0.48166037, 0.57330743, 0.06621459, 0.3704664],
                  [0.99777641, 0.87790733, 0.67211584, 0.46130919]])
dataY = np.array([0, 1])


# create NoisyLabels instance
test_data_noisy_labels = NoisyLabels("testdata", dataX, dataY, rules, ClassLabels, num_classes = 2)

# get noisy labels and confidence matrices 
L,S = test_data_noisy_labels.get_labels()

# create the label number to label name mapping json
test_data_noisy_labels.generate_json("enumjson")

# checking
with open("enumjson", 'r') as f:
    json_object = json.load(f)
print(json_object)

# create pickle file containing noisy labels
test_data_noisy_labels.generate_pickle("enumpkl")

# checking
f=open("enumpkl","rb")
noisy_data = pickle.load(f)
print(noisy_data["l"])
print(noisy_data["s"])