from labeling.lf import *
from labeling.apply import *
from labeling.preprocess import *
from labeling.continuous_scoring import *
from labeling.noisy_labels import *
from labeling.lf_set import *
import numpy as np

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

@labeling_function(pre=[square], resources=lf_resources, cont_scorer=score, label=0)
def lf1(x, **kwargs):
    if np.linalg.norm(x['value']) < 1 and kwargs["r3"]==4:
        if (kwargs['continuous_score']>0.01):                       # can use continuous score inside now
            return 0
    return -1

@labeling_function(pre=[square], label=1)                # no continuous scorer specified
def lf2(x, **kwargs):
    if np.linalg.norm(x['value']) < 1:
        return 1
    return -1

## creating a RuleSet object with desired Labeling functions ##
lfs = [lf1, lf2]
# rules = LFSet("myrules", lfs)
rules = LFSet("testrules")
rules.add_lf_list(lfs)
# rules.add_lf(lf1)
# rules.add_lf(lf2)

## Data ##
dataX = np.array([[0.7659027,  0.07041862, 0.67856597, 0.58097795],
[0.98964838, 0.29277118, 0.67217224, 0.69125625],
[0.25344225, 0.72530643, 0.52627362, 0.08560926]])
dataY = np.array([0, 1, 1])

## Creating NoisyLabels class ##
test_data_noisy_labels = NoisyLabels("testdata", dataX, dataY, rules)

## Getting Noisy Labels ##
L,S = test_data_noisy_labels.get_labels()

## Generating pickle file ##
test_data_noisy_labels.generate_pickle()

# Checking correctness ##
if (next(iter(rules.get_lfs())) == lf1):
    Lc=np.array([[0,  1],[-1, -1],[ 0,  1]])
    Sc=np.array([[0.26463369, -1.],[-1.,-1.],[ 0.32993693, -1.]])
else:
    Lc=np.array([[1,  0],[-1, -1],[1,  0]])
    Sc=np.array([[-1., 0.26463369],[-1.,-1.],[-1., 0.32993693]])



f=open("testdata_pickle","rb")
loaded=[]
for i in range(9):
    loaded+=[pickle.load(f)]

if np.allclose(Lc, loaded[1]) and np.allclose(Sc, loaded[6]):
    print("works fine")
else:
    print("something went wrong")