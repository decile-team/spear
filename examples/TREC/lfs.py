import numpy as np
import re
import enum

import sys
sys.path.append('../../')

from spear.labeling import labeling_function, LFSet, ABSTAIN, preprocessor

from preprocessor import convert_to_lower

label_map = {"DESC": "DESCRIPTION",
            "ENTY": "ENTITY",
            "HUM": "HUMAN",
            "ABBR": "ABBREVIATION",
            "LOC": "LOCATION",
            "NUM": "NUMERIC"}

class ClassLabels(enum.Enum):
    DESCRIPTION     = 0
    ENTITY          = 1
    HUMAN           = 2
    ABBREVIATION    = 3
    LOCATION        = 4
    NUMERIC         = 5


def load_rules(file_name='rules.txt'):
    rules = LFSet("TREC_LFS")
    
    with open(file_name, 'r', encoding='latin1') as f:
        i = 0
        for line in f:
            list_in = line.strip().split("\t")
            label = ClassLabels[label_map[list_in[0]]]
            pattern = list_in[1]
            rule_name = "rule"+str(i)
            
            @labeling_function(name=rule_name,resources=dict(pattern=pattern,output=label),pre=[convert_to_lower],label=label)
            def f(x,**kwargs):
                result = re.findall(kwargs["pattern"], x)
                if result:
                    return kwargs["output"]
                else:
                    return ABSTAIN

            rules.add_lf(f)
            i = i+1
    return rules

rules = load_rules()
