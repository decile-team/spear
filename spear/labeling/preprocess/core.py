from typing import Any, Callable, List, Mapping, Optional

from ..data_types import DataPoint


class BasePreprocessor:
    def __init__(
        self,
        name: str,
        f: Callable[..., int],
        resources: Optional[Mapping[str, Any]] = None,
        # pre: Optional[List[BasePreprocessor]] = None,
    ) -> None:
        self.name = name
        self._f = f
        self._resources = resources or {}
    
    def __call__(self, x: DataPoint) -> DataPoint:
        return self._f(x,**self._resources)

    def __repr__(self) -> str:
        return f"{type(self).__name__} {self.name}"

class preprocessor:
    def __init__(
        self,
        name: Optional[str] = None,
        resources: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if callable(name):
            raise ValueError("Looks like this decorator is missing parentheses!")
        self.name = name
        self.resources = resources

    def __call__(self, f: Callable[..., int]) -> BasePreprocessor:
        name = self.name or f.__name__
        return BasePreprocessor(name=name, resources=self.resources, f=f)
