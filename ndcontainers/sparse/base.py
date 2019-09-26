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

    @abc.abstractmethod
    def reshape(self, *shape, copy=True, order='C'):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def nbytes(self):
        raise NotImplementedError

    @nbytes.setter
    def nbytes(self, value):
        self._setter_not_writeable('nbytes')

    @property
    def dtype(self):
        return self.data.dtype

    @dtype.setter
    def dtype(self, value):
        if not np.can_cast(self.fill_value, value):
            raise ValueError(f"'fill_value' {self.fill_value!r} cannot be cast to {value!r}")
        self.data.dtype = np.dtype(value)

    @property
    def fill_value(self):
        return self._fill_value

    @fill_value.setter
    def fill_value(self, value):
        if not np.can_cast(value, self.dtype):
            raise ValueError(f"'fill_value' {value!r} cannot be cast to {self.dtype!r}")
        self._fill_value = value

    @property
    def ndim(self):
        return len(self.shape)
    
    @ndim.setter
    def ndim(self, value):
        self._setter_not_writeable('ndim')

    @property
    def nnz(self):
        return len(self.data)
    
    @nnz.setter
    def nnz(self, value):
        self._setter_not_writeable('nnz')

    @property
    def size(self):
        return np.prod(self.shape, dtype=int)

    @size.setter
    def size(self, value):
        self._setter_not_writeable('size')

    def _setter_not_writeable(self, name):
        raise AttributeError(f"attribute '{name}' of '{type(self).__name__}' objects is not writeable")
