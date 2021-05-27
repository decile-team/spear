from labeling.lf import *
from labeling.apply import *
from labeling.preprocess import *
from labeling.analysis import *
import numpy as np

@preprocessor()
def square(x):
    return {"value":x*x}


@labeling_function(pre=[square])
def func(x):
    if x['value'] == 0:
        return 0
    else:
        return 1


lfs = [func]

[print(x) for x in lfs]

data = np.ones((5,1))
applier = LFApplier(lfs=lfs)
L,S=applier.apply(data)
if np.allclose(L,np.ones((5,1))):
    print("="*10+"Basic lf and apply testing is successfull"+"="*10)
else:
    print("="*10+"Something went wrong"+"="*10)


analysis = LFAnalysis(L,lfs)

analysis.lf_summary()
