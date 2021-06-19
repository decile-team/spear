import enum
import numpy as np
from tqdm import tqdm
from itertools import chain
from typing import DefaultDict, Dict, List, Set, NamedTuple, Tuple, Union

from ..lf_set import LFSet
from ..utils import check_unique_names
from ..lf import ABSTAIN, LabelingFunction
from ..data_types import DataPoint, DataPoints


RowData = List[Tuple[int, int, int, float]]     # index of datapoint, index of lf, label, confidence


class ApplierMetadata(NamedTuple):
    """Metadata about Applier call."""

    # Map from LF name to number of faults in apply call
    faults: Dict[str, int]


class _FunctionCaller:
    def __init__(self, fault_tolerant: bool):
        self.fault_tolerant = fault_tolerant
        self.fault_counts: DefaultDict[str, int] = DefaultDict(int)

    def __call__(self, f: LabelingFunction, x: DataPoint):
        if not self.fault_tolerant:
            return f(x)
        try:
            return f(x)
        except Exception:
            self.fault_counts[f.name] += 1
            return -1


class BaseLFApplier:
    """Base class for LF applier objects.
    Base class for LF applier objects, which executes a set of LFs
    on a collection of data points. Subclasses should operate on
    a single data point collection format (e.g. ``DataFrame``).
    Subclasses must implement the ``apply`` method.

    Args:
        lf_set (LFSet): Instace of LFset which has information of set of labeling functions(which is applied on data)
    
    Raises:
        ValueError:
            If names of LFs are not unique
    """
    _use_recarray = False

    def __init__(self, lf_set: LFSet) -> None:    
        # self._lf_set = lf_set
        self._lfs = lf_set.get_lfs()
        self._lf_names = [lf.name for lf in lf_set.get_lfs()]
        # self._enum = enum
        check_unique_names(self._lf_names)

    def _numpy_from_row_data(self, labels: List[RowData]) -> np.ndarray:
        L = np.empty((len(labels), len(self._lfs)), dtype=object)
        L.fill(ABSTAIN)
        S = np.full((len(labels), len(self._lfs)), None) #np.zeros((len(labels), len(self._lfs)), dtype=float) - 1.0
        # NB: this check will short-circuit, so ok for large L
        if any(map(len, labels)):
            row, col, enm, conf = zip(*chain.from_iterable(labels))
            L[row, col] = enm
            S[row, col] = conf

        if self._use_recarray:                                        # always false
            n_rows, _ = L.shape
            dtype = [(name, np.int64) for name in self._lf_names]
            recarray = np.recarray(n_rows, dtype=dtype)
            for idx, name in enumerate(self._lf_names):
                recarray[name] = L[:, idx]

            return recarray
        else:
            return L,S

    def __repr__(self) -> str:
        return f"{type(self).__name__}, LFs: {self._lf_names}"


def apply_lfs_to_data_point(
    x: DataPoint, index: int, lfs: List[LabelingFunction], f_caller: _FunctionCaller
) -> RowData:
    """Label a single data point with a set of LFs

    Args:
        x (DataPoint): Data point to label
        index (int): Index of the data point
        lfs (List[LabelingFunction]): List of LFs to label ``x`` with
        f_caller (_FunctionCaller): A ``_FunctionCaller`` to record failed LF executions

    Returns:
        RowData: A list of (data point index, LF index, label enum, confidence) tuples
    """
    labels = []
    for j, lf in enumerate(lfs):
        y, z = f_caller(lf, x)
        if (y==ABSTAIN and z is None):
            continue
        if (y==ABSTAIN and z is not None):
            labels.append((index, j, y, z))
            continue    
        assert(lf._label == y)
        labels.append((index, j, y.value, z))
    return labels


class LFApplier(BaseLFApplier):
    """LF applier for a list of data points (e.g. ``SimpleNamespace``) or a NumPy array.

    Args:
        lf_set (LFSet): Instace of LFset which has information of set of labeling functions(which is applied on data)
    """ 

    # Example:
    #     >>> from labeling.lf import labeling_function
    #     >>> @labeling_function()
    #     ... def is_big_num(x,**kwargs):
    #     ...     return 1 if x.num > 42 else 0
    #     >>> applier = LFApplier([is_big_num])
    #     >>> from types import SimpleNamespace
    #     >>> applier.apply([SimpleNamespace(num=10), SimpleNamespace(num=100)])
    #     array([[0], [1]])
    #     >>> @labeling_function()
    #     ... def is_big_num_np(x):
    #     ...     return 1 if x[0] > 42 else 0
    #     >>> applier = LFApplier([is_big_num_np])
    #     >>> applier.apply(np.array([[10], [100]]))
    #     array([[0], [1]])  

    def apply(
        self,
        data_points: Union[DataPoints, np.ndarray],
        progress_bar: bool = True,
        fault_tolerant: bool = False,
        return_meta: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, ApplierMetadata]]:
        """Label list of data points or a NumPy array with LFs.

        Args:
            data_points (Union[DataPoints, np.ndarray]): List of data points or NumPy array to be labeled by LFs
            progress_bar (bool, optional): Display a progress bar?. Defaults to True.
            fault_tolerant (bool, optional): Output ``-1`` if LF execution fails?. Defaults to False.
            return_meta (bool, optional): Return metadata from apply call?. Defaults to False.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, ApplierMetadata]]:
                np.ndarray:
                    Matrix of labels emitted by LFs
                ApplierMetadata:
                    Metadata, such as fault counts, for the apply call
        """        
        labels = []
        f_caller = _FunctionCaller(fault_tolerant)
        if progress_bar:
            with tqdm(total=len(data_points)) as pbar:
                for i, x in enumerate(data_points):
                    labels.append(apply_lfs_to_data_point(x, i, self._lfs, f_caller))
                    pbar.update()
        else:
            for i, x in enumerate(data_points):
                labels.append(apply_lfs_to_data_point(x, i, self._lfs, f_caller))
        
        L,S = self._numpy_from_row_data(labels)
        if return_meta:
            return L, ApplierMetadata(f_caller.fault_counts)
        return L,S