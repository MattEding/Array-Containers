import abc

import numpy as np


__all__ = ['SparseArrayABC']

class SparseArrayABC(abc.ABC):
    @abc.abstractmethod
    def __array__(self):
        raise NotImplementedError 

    def __str__(self):
        return (
            f"<{type(self).__name__}: shape={self.shape}, nnz={self.nnz}, "
            f"dtype={self.dtype}, fill_value={self.fill_value}>"
        )

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def nnz(self):
        return len(self.data)

    @property
    def size(self):
        return np.prod(self.shape, dtype=int)
