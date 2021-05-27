import numpy as np
import pickle, enum, json
from typing import Any, Callable, List, Mapping, Optional

from ..lf_set import LFSet
from ..apply import LFApplier
from ..analysis import LFAnalysis
from ..lf import LabelingFunction
from ..utils import dump_to_pickle
from ..data_types import DataPoint, DataPoints


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
        rules: LFSet,
        gold_labels: Optional[DataPoints] = np.array([]),
        labels_enum = None,
        num_classes = -1,
        exemplars: DataPoints=np.array([]),
    ) -> None:       
        """Instantiates NoisyLabels class with dataset and set of LFs to noisily label the dataset
        """
        self.name = name
        self._data = data
        self._gold_labels = gold_labels
        self._rules = rules
        self._labels_enum = labels_enum
        self._num_classes = num_classes
        self._L = None
        self._S = None
        self._R = exemplars
        self._enum = enum

        assert(num_classes == len(labels_enum))

    def get_labels(self):
        """Applies LFs to the dataset to generate noisy labels and returns noisy labels and confidence scores

        Returns:
            Tuple(DataPoints, DataPoints): Noisy Labels and Confidences
        """
        if self._L is None or self._L is None:
            applier = LFApplier(lf_set = self._rules)
            L,S = applier.apply(self._data)
            self._L = L
            self._S = S
        return self._L, self._S

    def analyse_lfs(self,plot=False):
        """Analyse the lfs in LFSet on data

        Args:
            plot (bool, optional): Plot the values. Defaults to False.

        Returns:
            DataFrame: dataframe consisting of Ploarity, Coverage, Overlap, Conflicts, Empirical Acc
        """        
        if self._L is None or self._L is None:
            applier = LFApplier(lf_set = self._rules)
            L,S = applier.apply(self._data)
            self._L = L
            self._S = S
        
        analysis = LFAnalysis(self._labels_enum,self._L,self._rules)
        if len(self._gold_labels) == 0:
            df = analysis.lf_summary(plot=plot)
        else:
            df = analysis.lf_summary(self._gold_labels,plot=plot)
        return df
        

    def generate_json(self, filename=None):
        """Generates a json file with label value to label name mapping

        Args:
            filename (str, optional): Name for json file. Defaults to None.
        """
        if filename is None:
            filename = self.name+"_json.json"
        
        dic = {}
        for e in self._labels_enum:
            dic[e.value]=e.name

        with open(filename, "w") as outfile:
            json.dump(dic, outfile)
        
    def generate_pickle(self, filename=None):
        """Generates a pickle file with noisy labels, confidence and other Metadata

        Args:
            filename (str, optional): Name for pickle file. Defaults to None.
        """
        if filename is None:
            filename = self.name+"_pickle"
        
        if (self._L is None or self._S is None):
            applier = LFApplier(lf_set = self._rules)
            L,S = applier.apply(self._data)
            self._L = L
            self._S = S

        num_inst=self._data.shape[0]
        num_rules=self._L.shape[1]

        x=self._data
        l=self._L

        m=(self._L!=ABSTAIN).astype(int)                                        # lf covers example or not 
        L=self._gold_labels                                                     # true labels
        d=np.ones((num_inst, 1))                                                # belongs to labeled data or not
        r=self._R                                                               # exemplars

        s=self._S                                                               # continuous scores
        n=np.array([lf._is_cont for lf in self._rules.get_lfs()], dtype=bool)   # lf continuous or not
        k=np.array([lf._label.value for lf in self._rules.get_lfs()])           # lf associated to which class

        output = dict()
        output["x"] = x
        output["l"] = l
        output["m"] = m
        output["L"] = L
        output["d"] = d
        output["r"] = r
        output["s"] = s
        output["n"] = n
        output["k"] = k
        output["num_classes"] = self._num_classes
        to_dump = [output]

        dump_to_pickle(filename, to_dump)






