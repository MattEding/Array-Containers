import abc

import numpy as np

from . import Array


__all__ = ['SparseArray']


class SparseArray(Array):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # ABSTRACT

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # PROPERTIES

    @property
    def data(self):
        """A read-only view of the underlying 'data' array."""
        view = self._data.view()
        view.setflags(write=False)
        return view

    @property
    def idxs(self):
        """A read-only view of the underlying 'idxs' array."""
        view = self._idxs.view()
        view.setflags(write=False)
        return view

    @property
    def fill_value(self):
        """The value of unspecified elements for this array."""
        return self._fill_value

    @fill_value.setter
    def fill_value(self, value):
        if not np.can_cast(value, self.dtype):
            raise ValueError(f"'fill_value' {value!r} cannot be cast to {self.dtype!r}")
        self._fill_value = value

    @property
    def nnz(self):
        """The number of "non-zero" elements."""
        return len(self.data)

    @nnz.setter
    def nnz(self, value):
        self._setter_not_writeable('nnz')

    @property
    def sparsity(self):
        """TODO"""
        return self.nnz / self.size

    @sparsity.setter
    def sparsity(self):
        self._setter_not_writeable('sparsity')

    # - - - - - - - - - - Superclass Property Overrides - - - - - - - - - -

    @Array.dtype.getter
    def dtype(self):
        return self.data.dtype

    @dtype.setter
    def dtype(self, value):
        if not np.can_cast(self.fill_value, value):
            raise ValueError(f"'fill_value' {self.fill_value!r} cannot be cast to {value!r}")
        dtype = np.dtype(value)
        if dtype.itemsize != self.idxs.itemsize:
            raise ValueError(f"'data' {dtype!r} does not preserve shape with 'idxs'")
        self._data.dtype = dtype

    @Array.shape.getter
    def shape(self):
        return self._shape

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # DUNDERS

    def __bool__(self):
        super().__bool__()
        ...

    def __str__(self):
        return (
            f"<{type(self).__name__}: shape={self.shape}, nnz={self.nnz},"
            f" dtype={self.dtype}, fill_value={self.fill_value}>"
        )
