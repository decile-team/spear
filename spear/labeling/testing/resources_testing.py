from labeling.lf import *
from labeling.apply import *
from labeling.preprocess import *
from labeling.continuous_scoring import *
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
        return 0
    return -1

@labeling_function(pre=[square], label=1)                # no continuous scorer specified
def lf2(x, **kwargs):
    if np.linalg.norm(x['value']) < 1:
        return 1
    return -1

lfs = [lf1, lf2]
rules = LFSet("myrules")
rules.add_lf_list(lfs)

dataX = np.array([[0.14912444, 0.83544616, 0.61849807, 0.43523642],
                 [0.14795163, 0.9986555,  0.27234144, 0.87403315]])
dataY = np.array([0, 1])
applier = LFApplier(lf_set=rules)
L,S=applier.apply(dataX)

if (next(iter(rules.get_lfs())) == lf1):
    Lc=np.array([[0,  1],[-1, -1]])
    Sc=np.array([[0.2646661, -1.],[-1.,-1.]])
else:
    Lc=np.array([[1,  0],[-1, -1]])
    Sc=np.array([[-1., 0.2646661],[-1.,-1.]])

if (np.allclose(S,Sc) and np.allclose(L,Lc)):
    print("works fine")
else:
    print("something went wrong")

