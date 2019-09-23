import abc
import inspect
# import numpy as np


class BaseSparse(abc.ABC):
    def __repr__(self):
        # use inspect.signature to check which parameters are ndarrays
        # and then dynamically generate the 
        return f"{type(self).__class__}"
    
    @abc.abstractmethod
    def __array__(self):
        return 