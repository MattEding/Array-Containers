import abc

import numpy as np


__all__ = ['SparseArrayABC']

class SparseArrayABC(abc.ABC):
    @abc.abstractmethod
    def __array__(self):
        return 

    
    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return np.prod(self.shape, dtype=int)

    # @abc.abstractmethod
    # def reshape(self, shape, order='C'):
    #     pass
