import abc

import numpy as np


__all__ = ['SparseArray']


class SparseArray(abc.ABC):
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
        """Gives a new shape to a sparse array without changing its data.

        Parameters
        ----------
        shape : length-2 tuple of ints
            The new shape should be compatible with the original shape. If
            an integer, then the result will be a 1-D array of that length.
            One shape dimension can be -1. In this case, the value is
            inferred from the length of the array and remaining dimensions

        copy : bool, optional
            Indicates whether or not attributes of self should be copied
            whenever possible. The degree to which attributes are copied varies
            depending on the type of sparse array being used.

        order : {'C', 'F'}, optional
            Read the elements using this index order. 'C' means to read and
            write the elements using C-like index order; e.g. read entire first
            row, then second row, etc. 'F' means to read and write the elements
            using Fortran-like index order; e.g. read entire first column, then
            second column, etc.

        Returns
        -------
        reshaped_array : SparseArray
            A sparse array with the given `shape`, not necessarily of the same
            format as the current object.`
        """
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
        #FIXME: with self._data now, should this be able to be set?
        #       I made self.data read-only, but should I make self._data.dtype a thing?
        #       or should I just have an astype(type, copy=...)?
        self.data.dtype = np.dtype(value)

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
    def ndim(self):
        """The number of dimensions for this array."""
        return len(self.shape)
    
    @ndim.setter
    def ndim(self, value):
        self._setter_not_writeable('ndim')

    @property
    def nnz(self):
        """The number of "non-zero" elements."""
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

    @property
    def sparsity(self):
        return self.nnz / self.size

    @sparsity.setter
    def sparsity(self):
        self._setter_not_writeable('sparsity')

    #TODO: change to a descriptor to also encompass deleters
    def _setter_not_writeable(self, name):
        raise AttributeError(f"attribute '{name}' of '{type(self).__name__}' objects is not writeable")

    #TODO: how to approach allowing view-only of data and idxs
    @property
    def data(self):
        """A read-only view of the underlying data"""
        view = self._data.view()
        view.flags['WRITEABLE'] = False
        return view

    @property
    def idxs(self):
        """A read-only view of the underlying idxs"""
        view = self._idxs.view()
        view.flags['WRITEABLE'] = False
        return view

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self):
        self._setter_not_writeable('shape')