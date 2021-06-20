from typing import Any, Callable, List, Mapping, Optional

from ..data_types import DataPoint
from ..preprocess import BasePreprocessor
from ..continuous_scoring import BaseContinuousScorer

ABSTAIN = None

class LabelingFunction:
    """Base class for labeling function

    Args:
        name (str): name for this LF object
        f (Callable[..., int]): core function which labels the input
        label (enum): Which class this LF corresponds to
        resources (Optional[Mapping[str, Any]], optional): Additional resources for core function. Defaults to None.
        pre (Optional[List[BasePreprocessor]], optional): Preprocessors to apply on input before labeling. Defaults to None.
        cont_scorer (Optional[BaseContinuousScorer], optional): Continuous Scorer to calculate the confidence score. Defaults to None.
    """    
    def __init__(
        self,
        name: str,
        f: Callable[..., int],                              
        label = None,
        resources: Optional[Mapping[str, Any]] = None,
        pre: Optional[List[BasePreprocessor]] = None,
        cont_scorer: Optional[BaseContinuousScorer] = None,
    ) -> None:
        """Instatiates LabelingFunction class object
        """
        assert(label is not None)
        self.name = name
        self._f = f
        self._label = label
        self._resources = resources or {}
        self._pre = pre or []
        self._cont_scorer = cont_scorer
        if self._cont_scorer is None:
            self._is_cont=0
        else:
            self._is_cont=1

    def _preprocess_data_point(self, x: DataPoint) -> DataPoint:
        """Preprocesses input by applying each preprocessing function in succession

        Args:
            x (DataPoint): Single datapoint

        Raises:
            ValueError: When a preprocessing function returns None

        Returns:
            DataPoint: Preprocessed datapoint
        """
        for preprocessor in self._pre:
            x = preprocessor(x)
            if x is None:
                raise ValueError("Preprocessor should not return None")
        return x

    def __call__(self, x: DataPoint):                                                                           # -> (Enum, float)
        """Applies core labeling function and continuous scorer on datapoint and returns label and confidence

        Args:
            x (DataPoint): Datapoint 

        Returns:
            (Enum, float): Label enum object and confidence for the datapoint

        """
        x = self._preprocess_data_point(x)
        if self._is_cont:
            cs = self._cont_scorer(x,**self._resources)     # continuous score
            dic = {"continuous_score": cs}
            return self._f(x,**self._resources, **dic), cs
        else:
            cs = None
            return self._f(x,**self._resources), cs                                   
        
    def __repr__(self) -> str:
        """Represents class object as string

        Returns:
            str: string representation of the class object
        """
        preprocessor_str = f", Preprocessors: {self._pre}"
        return f"{type(self).__name__} {self.name}{preprocessor_str}"


class labeling_function:
    """Decorator class for a labeling function
    
    Args:
        name (Optional[str], optional): Name for this labeling function. Defaults to None.
        label (Optional[Enum], optional): An enum. Which class this LF corresponds to. Defaults to None.
        resources (Optional[Mapping[str, Any]], optional): Additional resources for the LF. Defaults to None.
        pre (Optional[List[BasePreprocessor]], optional): Preprocessors to apply on input before labeling . Defaults to None.
        cont_scorer (Optional[BaseContinuousScorer], optional): Continuous Scorer to calculate the confidence score. Defaults to None.

    Raises:
        ValueError: If the decorator is missing parantheses
    """    
    def __init__(
        self,
        name: Optional[str] = None,
        label = None,
        resources: Optional[Mapping[str, Any]] = None,
        pre: Optional[List[BasePreprocessor]] = None,
        cont_scorer: Optional[BaseContinuousScorer] = None,
    ) -> None:
        """Instatiates decorator for labeling function
        """
        if callable(name):
            raise ValueError("Looks like this decorator is missing parentheses!")
        self.name = name
        self.label = label
        self.resources = resources
        self.pre = pre
        self.cont_scorer = cont_scorer

    def __call__(self, f: Callable[..., int]) -> LabelingFunction:
        """Creates and returns a LabelingFunction object for labeling Datapoint

        Args:
            f (Callable[..., int]): core function which labels the input

        Returns:
            LabelingFunction: a callable LabelingFunction object 
        """
        name = self.name or f.__name__
        return LabelingFunction(name=name, resources=self.resources, f=f, pre=self.pre, cont_scorer=self.cont_scorer, label=self.label)
