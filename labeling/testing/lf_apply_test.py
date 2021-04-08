from labeling.lf import *
from labeling.apply import *
from labeling.preprocess import *
import numpy as np

@preprocessor()
def square(x, **kwargs):
    return {"value":x*x}


@labeling_function(pre=[square], label=0)
def func(x, **kwargs):
    if x['value'] == 0:
        return 0
    else:
        return 1

@labeling_function(pre=[square])
def func1(x):
    if x['value'] == 0:
        return 0
    else:
        return 1

lfs = [func,func1]
rules = LFSet("myrules")
rules.add_lf_list(lfs)

[print(x) for x in lfs]

data = np.ones((5,1))
applier = LFApplier(lf_set=rules)
L,S=applier.apply(data)
if np.allclose(L,np.ones((5,2))):
    print("="*10+"Basic lf and apply testing is successfull"+"="*10)
else:
    print("="*10+"Something went wrong"+"="*10) 