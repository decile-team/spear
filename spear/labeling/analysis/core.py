import warnings
import numpy as np
from itertools import product
import scipy.sparse as sparse
from collections import OrderedDict
from pandas import DataFrame, Series
from typing import List, Optional, Union
from sklearn.metrics import confusion_matrix

from ..utils import plot_df_bar
from ..lf import LabelingFunction

warnings.simplefilter(action='ignore', category=FutureWarning)

class LFAnalysis:   
    """Run analysis on LFs using label matrix.

        Args:
            L (np.ndarray): Label matrix where L_{i,j} is the label given by the jth LF to the ith x instance
            lfs (Optional[List[LabelingFunction]], optional): Labeling functions used to generate `'L``. Defaults to None.
            abstain (int, optional): label associated with abstain. Defaults to -1.

        Raises:
            ValueError:  If number of LFs and number of LF matrix columns differ    
    """

    def __init__(
        self, enum, L: np.ndarray, rules = None
    ) -> None:  
        self.L,self.mapping = self._create_L(enum,L)
        self._L_sparse = sparse.csr_matrix(self.L + 1)
        self._lf_names = None
        lfs = list(rules.get_lfs())
        if lfs is not None:
            if len(lfs) != self.L.shape[1]:
                raise ValueError(
                    f"Number of LFs ({len(lfs)}) and number of "
                    f"LF matrix columns ({self.L.shape[1]}) are different"
                )
            self._lf_names = [lf.name for lf in lfs]

    def _create_L(self,enum,L):
        """Map the enum values to non-ve integers and abstain to -1"""        
        mapping = {}
        j = 0
        L_num = -1 * np.ones_like(L,dtype=float)
        for i in enum:
            mapping[i.value] = j
            L_num[L==i.value] = j
            j = j+1
        return L_num, mapping
        


    def _covered_data_points(self) -> np.ndarray:
        """Get indicator vector z where z_i = 1 if x_i is labeled by at least one LF."""
        return np.ravel(np.where(self._L_sparse.sum(axis=1) != 0, 1, 0))
    
    def _overlapped_data_points(self) -> np.ndarray:
        """Get indicator vector z where z_i = 1 if x_i is labeled by more than one LF."""
        return np.where(np.ravel((self._L_sparse != 0).sum(axis=1)) > 1, 1, 0)

    
    def _conflicted_data_points(self) -> np.ndarray:
        """Get indicator vector z where z_i = 1 if x_i is labeled differently by two LFs."""
        m = sparse.diags(np.ravel(self._L_sparse.max(axis=1).todense()))
        return np.ravel(
            np.max(m @ (self._L_sparse != 0) != self._L_sparse, axis=1)
            .astype(int)
            .todense()
        )

    
    def label_coverage(self) -> float:
        """Compute the fraction of data points with at least one label.

        Returns:
            float: Fraction of data points with labels
        
        Example:
            >>> L = np.array([
            ...     [-1, 0, 0],
            ...     [-1, -1, -1],
            ...     [1, 0, -1],
            ...     [-1, 0, -1],
            ...     [0, 0, 0],
            ... ])
            >>> LFAnalysis(L).label_coverage()
            0.8
        """
        return self._covered_data_points().sum() / self.L.shape[0]
    

    def label_overlap(self) -> float:
        """Compute the fraction of data points with at least two (non-abstain) labels.

        Returns:
            float: Fraction of data points with overlapping labels
        
        Example:
            >>> L = np.array([
            ...     [-1, 0, 0],
            ...     [-1, -1, -1],
            ...     [1, 0, -1],
            ...     [-1, 0, -1],
            ...     [0, 0, 0],
            ... ])
            >>> LFAnalysis(L).label_overlap()
            0.6  
        """
        return self._overlapped_data_points().sum() / self.L.shape[0]
    

    def label_conflict(self) -> float:
        """Compute the fraction of data points with conflicting (non-abstain) labels.

        Returns:
            float: Fraction of data points with conflicting labels
        
        Example:
            >>> L = np.array([
            ...     [-1, 0, 0],
            ...     [-1, -1, -1],
            ...     [1, 0, -1],
            ...     [-1, 0, -1],
            ...     [0, 0, 0],
            ... ])
            >>> LFAnalysis(L).label_conflict()
            0.2
        """        
        return self._conflicted_data_points().sum() / self.L.shape[0]
    
    def lf_polarities(self) -> List[List[int]]:
        """Infer the polarities of each LF based on evidence in a label matrix.

        Returns:
            List[List[int]]: Unique output labels for each LF
        
        Example:
            >>> L = np.array([
            ...     [-1, 0, 0],
            ...     [-1, -1, -1],
            ...     [1, 0, -1],
            ...     [-1, 0, -1],
            ...     [0, 0, 0],
            ... ])
            >>> LFAnalysis(L).lf_polarities()
            [[0, 1], [0], [0]]
        """
        return [
            sorted(list(set(self.L[:, i])))
            for i in range(self.L.shape[1])
        ]
    

    def lf_coverages(self) -> np.ndarray:
        """Compute frac. of examples each LF labels.

        Returns:
            np.ndarray: Fraction of labeled examples for each LF
        
        Example:
            >>> L = np.array([
            ...     [-1, 0, 0],
            ...     [-1, -1, -1],
            ...     [1, 0, -1],
            ...     [-1, 0, -1],
            ...     [0, 0, 0],
            ... ])
            >>> LFAnalysis(L).lf_coverages()
            array([0.4, 0.8, 0.4])
        """
        return np.ravel((self.L != -1).sum(axis=0)) / self.L.shape[0]
    

    def lf_overlaps(self, normalize_by_coverage: bool = False) -> np.ndarray:
        """Compute frac. of examples each LF labels that are labeled by another LF.
        An overlapping example is one that at least one other LF returns a
        (non-abstain) label for.
        Note that the maximum possible overlap fraction for an LF is the LF's
        coverage, unless ``normalize_by_coverage=True``, in which case it is 1

        Args:
            normalize_by_coverage (bool, optional): Normalize by coverage of the LF,
                                                    so that it returns the percent of LF labels that have overlaps.
                                                    Defaults to False.

        Returns:
            np.ndarray: Fraction of overlapping examples for each LF
        
        Example:
            >>> L = np.array([
            ...     [-1, 0, 0],
            ...     [-1, -1, -1],
            ...     [1, 0, -1],
            ...     [-1, 0, -1],
            ...     [0, 0, 0],
            ... ])
            >>> LFAnalysis(L).lf_overlaps()
            array([0.4, 0.6, 0.4])
            >>> LFAnalysis(L).lf_overlaps(normalize_by_coverage=True)
            array([1.  , 0.75, 1.  ])
        """
        overlaps = (
            (self._L_sparse != 0).T
            @ self._overlapped_data_points()
            / self._L_sparse.shape[0]
        )
        if normalize_by_coverage:
            overlaps /= self.lf_coverages()
        return np.nan_to_num(overlaps)
    

    def lf_conflicts(self, normalize_by_overlaps: bool = False) -> np.ndarray:
        """Compute frac. of examples each LF labels and labeled differently by another LF.
        A conflicting example is one that at least one other LF returns a
        different (non-abstain) label for.
        Note that the maximum possible conflict fraction for an LF is the LF's
        overlaps fraction, unless ``normalize_by_overlaps=True``, in which case it is 1.
        Parameters

        Args:
            normalize_by_overlaps (bool, optional): Normalize by overlaps of the LF, so that it returns the percent of LF
                                                    overlaps that have conflicts.
                                                    Defaults to False.

        Returns:
            np.ndarray: Fraction of conflicting examples for each LF
        
        Example:
            >>> L = np.array([
            ...     [-1, 0, 0],
            ...     [-1, -1, -1],
            ...     [1, 0, -1],
            ...     [-1, 0, -1],
            ...     [0, 0, 0],
            ... ])
            >>> LFAnalysis(L).lf_conflicts()
            array([0.2, 0.2, 0. ])
            >>> LFAnalysis(L).lf_conflicts(normalize_by_overlaps=True)
            array([0.5       , 0.33333333, 0.        ])
        """        
        conflicts = (
            (self._L_sparse != 0).T
            @ self._conflicted_data_points()
            / self._L_sparse.shape[0]
        )
        if normalize_by_overlaps:
            conflicts /= self.lf_overlaps()
        return np.nan_to_num(conflicts)
    

    def lf_empirical_accuracies(self, Y: np.ndarray) -> np.ndarray:
        """Compute empirical accuracy against a set of labels Y for each LF.
        Usually, Y represents development set labels.

        Args:
            Y (np.ndarray): [n] np.ndarray of gold labels

        Returns:
            np.ndarray: Empirical accuracies for each LF
        """
        X = np.where(
            self.L == -1,
            0,
            np.where(self.L == np.vstack([Y] * self.L.shape[1]).T, 1, -1),
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.nan_to_num(0.5 * (X.sum(axis=0) / (self.L != -1).sum(axis=0) + 1))
    

    # def lf_empirical_probs(self, Y: np.ndarray, k: int) -> np.ndarray:
    #     """Estimate conditional probability tables for each LF.
    #     Computes conditional probability tables, P(L | Y), for each LF using
    #     the provided true labels Y.

    #     Args:
    #         Y (np.ndarray):  The n-dim array of true labels
    #         k (int): The cardinality i.e. number of classes

    #     Returns:
    #         np.ndarray: An m x (k+1) x k np.ndarray representing the m (k+1) x k conditional probability
    #         tables P_i, where P_i[l,y] represents P(LF_i = l | Y = y) empirically calculated
    #     """        
    #     n, m = self.L.shape

    #     Y = Y#to_int_label_array(Y)

    #     # Compute empirical conditional probabilities
    #     # Note: Can do this more efficiently...
    #     P = np.zeros((m, k + 1, k))
    #     for y in range(k):
    #         is_y = np.where(Y == y, 1, 0)
    #         for j, l in product(range(m), range(-1, k)):
    #             P[j, l + 1, y] = np.where(self.L[:, j] == l, 1, 0) @ is_y / is_y.sum()
    #     return P
    

    def lf_summary(
        self, Y: Optional[np.ndarray] = None, plot: Optional[bool] = False
    ) -> DataFrame:
        """Create a pandas DataFrame with the various per-LF statistics.

        Args:
            Y (Optional[np.ndarray], optional): [n] np.ndarray of gold labels.
                                                If provided, the empirical accuracy for each LF will be calculated.
                                                Defaults to None.
            plot (Optional[bool], optional): If set to true a bar graph is plotted. Defaults to False.

        Returns:
            DataFrame: Summary statistics for each LF
        """
        n, m = self.L.shape
        lf_names: Union[List[str], List[int]]
        d: OrderedDict[str, Series] = OrderedDict()
        if self._lf_names is not None:
            lf_names = self._lf_names
        else:
            lf_names = list(range(m))

        # Remap the true labels values
        print('Y is ', Y)
        if Y is not None:
            Y = np.array([self.mapping[v] for v in Y])

        # Default LF stats
        d["Polarity"] = Series(data=self.lf_polarities(), index=lf_names)
        d["Coverage"] = Series(data=self.lf_coverages(), index=lf_names)
        d["Overlaps"] = Series(data=self.lf_overlaps(), index=lf_names)
        d["Conflicts"] = Series(data=self.lf_conflicts(), index=lf_names)

        if Y is not None:
            labels = np.unique(
                np.concatenate((Y.flatten(), self.L.flatten(), np.array([-1])))
            )
            confusions = [
                confusion_matrix(Y, self.L[:, i], labels=labels)[1:, 1:] for i in range(m)
            ]
            corrects = [np.diagonal(conf).sum() for conf in confusions]
            incorrects = [
                conf.sum() - correct for conf, correct in zip(confusions, corrects)
            ]
            accs = self.lf_empirical_accuracies(Y)
            d["Correct"] = Series(data=corrects, index=lf_names)
            d["Incorrect"] = Series(data=incorrects, index=lf_names)
            d["Emp. Acc."] = Series(data=accs, index=lf_names)
        
        data_frame = DataFrame(data=d, index=lf_names)
        
        if plot == True:
            plot_df_bar(data_frame,"seperate")

        return data_frame
    