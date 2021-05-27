from typing import Any, Callable, List, Mapping, Optional

from ..data_types import DataPoint


class BaseContinuousScorer:
    """Base Class for Continuous Scoring function used by the Labeling Function

    Args:
        name (str): Name of the continuous scoring function
        cf (Callable[..., int]): Core function which calculates continuous score
        resources (Optional[Mapping[str, Any]], optional): Resources for the scorer. Defaults to None.
    """
    def __init__(
        self,
        name: str,
        cf: Callable[..., int],
        resources: Optional[Mapping[str, Any]] = None,
        # pre: Optional[List[BasePreprocessor]] = None,
    ) -> None:
        """Instantiate the BaseContinuousScorer Class
        """
        self.name = name
        self._cf = cf
        self._resources = resources or {}
    
    def __call__(self, x: DataPoint, **kwargs) -> float:
        """Applies core function on datapoint to give continuous score

        Args:
            x (DataPoint): Datapoint 

        Returns:
            float: continuous score output by the function
        """
        return self._cf(x,**self._resources, **kwargs)

    def __repr__(self) -> str:
        """Represents class objects as string

        Returns:
            str: representation of class object as string
        """
        return f"{type(self).__name__} {self.name}"

class continuous_scorer:
    """Decorator class for continuous scoring.

    Args:
        name (Optional[str], optional): Name for the decorator. Defaults to None.
        resources (Optional[Mapping[str, Any]], optional): Resources for the scorer. Defaults to None.

    Raises:
        ValueError: If decorator is missing parantheses.
    """    
    def __init__(
        self,
        name: Optional[str] = None,
        resources: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Instantiates decorator for continuous scorer
        """
        if callable(name):
            raise ValueError("Looks like this decorator is missing parentheses!")
        self.name = name
        self.resources = resources

    def __call__(self, cf: Callable[..., int]) -> BaseContinuousScorer:
        """Creates a callable BaseContinuosScorer object for applying scorer on datapoint.

        Args:
            cf (Callable[..., int]): Core function for calculating continuous score

        Returns:
            BaseContinuousScorer: a callable BaseContinuousScorer class object
        """
        name = self.name or cf.__name__
        return BaseContinuousScorer(name=name, resources=self.resources, cf=cf)
