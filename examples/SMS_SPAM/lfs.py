import numpy as np
import re
import enum

import sys
sys.path.append('../../')

from spear.labeling import labeling_function, LFSet, ABSTAIN, preprocessor

from preprocessor import convert_to_lower
from con_scorer import word_similarity

class ClassLabels(enum.Enum):
    SPAM = 1
    HAM = 0

THRESHOLD = 0.8

trigWord1 = {"free","credit","cheap","apply","buy","attention","shop","sex","soon","now","spam"}
trigWord2 = {"gift","click","new","online","discount","earn","miss","hesitate","exclusive","urgent"}
trigWord3 = {"cash","refund","insurance","money","guaranteed","save","win","teen","weight","hair"}
notFreeWords = {"toll","Toll","freely","call","meet","talk","feedback"}
notFreeSubstring = {"not free","you are","when","wen"}
firstAndSecondPersonWords = {"I","i","u","you","ur","your","our","we","us","youre"}
thirdPersonWords = {"He","he","She","she","they","They","Them","them","their","Their"}


@preprocessor()
def convert_to_lower(x):
    return x.lower().strip()


@labeling_function(resources=dict(keywords=trigWord1),pre=[convert_to_lower],label=ClassLabels.SPAM)
def LF1(c,**kwargs):    
    if len(kwargs["keywords"].intersection(c.split())) > 0:
        return ClassLabels.SPAM
    else:
        return ABSTAIN

@labeling_function(resources=dict(keywords=trigWord2),pre=[convert_to_lower],label=ClassLabels.SPAM)
def LF2(c,**kwargs):
    if len(kwargs["keywords"].intersection(c.split())) > 0:
        return ClassLabels.SPAM
    else:
        return ABSTAIN

@labeling_function(resources=dict(keywords=trigWord3),pre=[convert_to_lower],label=ClassLabels.SPAM)
def LF3(c,**kwargs):
    if len(kwargs["keywords"].intersection(c.split())) > 0:
        return ClassLabels.SPAM 
    else:
        return ABSTAIN

@labeling_function(resources=dict(keywords=notFreeWords),pre=[convert_to_lower],label=ClassLabels.HAM)
def LF4(c,**kwargs):
    if "free" in c.split() and len(kwargs["keywords"].intersection(c.split()))>0:
        return ClassLabels.HAM
    else:
        return ABSTAIN

@labeling_function(resources=dict(keywords=notFreeSubstring),pre=[convert_to_lower],label=ClassLabels.HAM)
def LF5(c,**kwargs):
    for pattern in kwargs["keywords"]:    
        if "free" in c.split() and re.search(pattern,c, flags= re.I):
            return ClassLabels.HAM
    return ABSTAIN

@labeling_function(resources=dict(keywords=firstAndSecondPersonWords),pre=[convert_to_lower],label=ClassLabels.HAM)
def LF6(c,**kwargs):
    if "free" in c.split() and len(kwargs["keywords"].intersection(c.split()))>0:
        return ClassLabels.HAM
    else:
        return ABSTAIN


@labeling_function(resources=dict(keywords=thirdPersonWords),pre=[convert_to_lower],label=ClassLabels.HAM)
def LF7(c,**kwargs):
    if "free" in c.split() and len(kwargs["keywords"].intersection(c.split()))>0:
        return ClassLabels.HAM
    else:
        return ABSTAIN

@labeling_function(label=ClassLabels.SPAM)
def LF8(c,**kwargs):
    if (sum(1 for ch in c if ch.isupper()) > 6):
        return ClassLabels.SPAM
    else:
        return ABSTAIN

# @labeling_function()
# def LF9(c,**kwargs):
#     return ClassLabels.HAM.value

@labeling_function(cont_scorer=word_similarity,resources=dict(keywords=trigWord1),pre=[convert_to_lower],label=ClassLabels.SPAM)
def CLF1(c,**kwargs):
    if kwargs["continuous_score"] >= THRESHOLD:
        return ClassLabels.SPAM
    else:
        return ABSTAIN

@labeling_function(cont_scorer=word_similarity,resources=dict(keywords=trigWord2),pre=[convert_to_lower],label=ClassLabels.SPAM)
def CLF2(c,**kwargs):
    if kwargs["continuous_score"] >= THRESHOLD:
        return ClassLabels.SPAM
    else:
        return ABSTAIN

@labeling_function(cont_scorer=word_similarity,resources=dict(keywords=trigWord3),pre=[convert_to_lower],label=ClassLabels.SPAM)
def CLF3(c,**kwargs):
    if kwargs["continuous_score"] >= THRESHOLD:
        return ClassLabels.SPAM
    else:
        return ABSTAIN

@labeling_function(cont_scorer=word_similarity,resources=dict(keywords=notFreeWords),pre=[convert_to_lower],label=ClassLabels.HAM)
def CLF4(c,**kwargs):
    if kwargs["continuous_score"] >= THRESHOLD:
        return ClassLabels.HAM
    else:
        return ABSTAIN

@labeling_function(cont_scorer=word_similarity,resources=dict(keywords=notFreeSubstring),pre=[convert_to_lower],label=ClassLabels.HAM)
def CLF5(c,**kwargs):
    if kwargs["continuous_score"] >= THRESHOLD:
        return ClassLabels.HAM
    else:
        return ABSTAIN

@labeling_function(cont_scorer=word_similarity,resources=dict(keywords=firstAndSecondPersonWords),pre=[convert_to_lower],label=ClassLabels.HAM)
def CLF6(c,**kwargs):
    if kwargs["continuous_score"] >= THRESHOLD:
        return ClassLabels.HAM
    else:
        return ABSTAIN

@labeling_function(cont_scorer=word_similarity,resources=dict(keywords=thirdPersonWords),pre=[convert_to_lower],label=ClassLabels.HAM)
def CLF7(c,**kwargs):
    if kwargs["continuous_score"] >= THRESHOLD:
        return ClassLabels.HAM
    else:
        return ABSTAIN

@labeling_function(cont_scorer=lambda x: 1-np.exp(float(-(sum(1 for ch in x if ch.isupper()))/2)),label=ClassLabels.SPAM)
def CLF8(c,**kwargs):
    if kwargs["continuous_score"] >= THRESHOLD:
        return ClassLabels.SPAM
    else:
        return ABSTAIN

# @labeling_function()
# def CLF9(c,**kwargs):
#     return ClassLabels.HAM


LFS = [LF1,
    LF2,
    LF3,
    LF4,
    LF5,
    LF6,
    LF7,
    LF8,
    CLF1,
    CLF2,
    CLF3,
    CLF4,
    CLF5,
    CLF6,
    CLF7,
    CLF8]

rules = LFSet("SPAM_LF")
rules.add_lf_list(LFS)