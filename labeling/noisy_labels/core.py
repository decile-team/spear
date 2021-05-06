from typing import Any, Callable, List, Mapping, Optional
from labeling.mtypes import DataPoint, DataPoints
from labeling.apply import *
from labeling.lf import *
from labeling.lf_set import *
from labeling.utils.pickle import *

import pickle
import numpy as np

class NoisyLabels:
    """Generate noisy lables, continuous score  from lf's applied on data  

    Args:
        name (str): Name for this object.
        data (DataPoints): Datapoints.
        gold_labels (Optional[DataPoints]): Labels for datapoints if available.
        rules (LFSet): Set of Rules to generate noisy labels for the dataset.
        exemplars (DataPoints): [description]
    """    
    def __init__(
        self,
        name: str,
        data: DataPoints,
        gold_labels: Optional[DataPoints],
        rules: LFSet,
        enum : Dict = {},
        exemplars: DataPoints=[],
    ) -> None:       
        """Instantiates NoisyLabels class with dataset and set of LFs to noisily label the dataset
        """
        self.name = name
        self._data = data
        self._gold_labels = gold_labels
        self._rules = rules
        # self._L = None
        self._E = None
        self._S = None
        self._R = exemplars
        self._enum = enum

    def get_labels(self):
        """Applies LFs to the dataset to generate noisy labels and returns noisy labels and confidence scores

        Returns:
            Tuple(DataPoints, DataPoints): Noisy Labels and Confidences
        """
        if self._E is None or self._S is none:
            applier = LFApplier(lf_set = self._rules)
            E,S = applier.apply(self._data)
            self._E = E
            self._S = S
        return self._E, self._S

    def generate_pickle(self, filename=None):
        """Generates a pickle file with noisy labels, confidence and other Metadata

        Args:
            filename (str, optional): Name for pickle file. Defaults to None.
        """
        if filename is None:
            filename = self.name+"_pickle"
        
        if (self._E is None or self._S is None):
            applier = LFApplier(lf_set = self._rules)
            E,S = applier.apply(self._data)
            self._E = E
            self._S = S

        num_inst=self._data.shape[0]
        num_rules=self._E.shape[1]

        x=self._data
        # l=self._L
        e=self._E

        m=(self._E!=ABSTAIN).astype(int)                                        # lf covers example or not 
        L=self._gold_labels                                                     # true labels
        d=np.ones((num_inst, 1))                                                # belongs to labeled data or not
        r=self._R                                                               # exemplars

        s=self._S                                                               # continuous scores
        n=np.array([lf._is_cont for lf in self._rules.get_lfs()], dtype=bool)   # lf continuous or not
        k=np.array([lf._label for lf in self._rules.get_lfs()], dtype=int)      # lf associated to which class

        output = dict()
        output["x"] = x
        output["e"] = e
        # output["l"] = l
        output["m"] = m
        output["L"] = L
        output["d"] = d
        output["r"] = r
        output["s"] = s
        output["n"] = n
        output["k"] = k
        to_dump = [output]

        dump_to_pickle(filename, to_dump)






