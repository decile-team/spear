from labeling.lf import *
from labeling.apply import *
from labeling.preprocess import *
from labeling.continuous_scoring import *
import numpy as np

@preprocessor()
def square(x):
    return {"value":x*x}

@continuous_scorer()
def score(x):
    return np.exp(-1*np.linalg.norm(x['value']))

@labeling_function(pre=[square], cont_scorer=score, label=0)
def lf1(x, **kwargs):
    if np.linalg.norm(x['value']) < 1:
        return 0
    return -1

@labeling_function(pre=[square], label=0)                # no continuous scorer specified
def lf2(x, **kwargs):
    if np.linalg.norm(x['value']) < 1:
        return 1
    return -1


lfs = [lf1, lf2]
rules = LFSet("myrules")
rules.add_lf_list(lfs)
data = np.array([[0.0983969,0.52830115, 0.90600643, 0.24581662], [0.80224391, 0.69694779, 0.2144578,  0.56402219]])

applier = LFApplier(lf_set=rules)
L,S=applier.apply(data)

if (next(iter(rules.get_lfs())) != lf1):
    Lc=np.array([[1, 0], [1, 0]])
    Sc=np.array([[-1., 0.41930488],[-1., 0.41977896]])
else:
    Lc=np.array([[0, 1], [0, 1]])
    Sc=np.array([[0.41930488, -1.],[0.41977896, -1.]])

if (np.allclose(S,Sc) and np.allclose(L,Lc)):
    print("works fine")
else:
    print("something went wrong")

