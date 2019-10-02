import itertools
import warnings
from numbers import Number

import numpy as np

from ..abc import SparseArray
from ..utils import NDArrayReprMixin
from ..utils import is_broadcastable
from ..utils import ravel_sparse_multi_index
from ..utils import unravel_sparse_index


__all__ = ['CoordinateArray']


class CoordinateArray(
    np.lib.mixins.NDArrayOperatorsMixin, NDArrayReprMixin, SparseArray
):
    # is there a real advantage to idxs *before* data? if so change order
    def __init__(
        self, data, idxs, shape=None, fill_value=None, dtype=None,
        copy=False, sum_duplicates=True
    ):
        multi_idx = np.atleast_2d(np.array(idxs, dtype=np.uint, copy=copy))
        if multi_idx.ndim != 2:
            raise ValueError(f"'idxs' must be 2d, not {multi_idx.ndim}d")

        try:
            min_shape = multi_idx.max(axis=1) + 1
        except ValueError:
            min_shape = ()

        if shape is None:
            self._shape = tuple(min_shape)
        else:
            shape = np.broadcast_arrays(shape).pop().astype(np.uint).ravel()
            if len(shape) != len(min_shape):
                raise ValueError(f"'shape' length does not match 'idxs' length")
            elif np.any(shape < min_shape):
                raise ValueError(f"'shape' {tuple(shape)} values are too small for 'idxs'")
            else:
                self._shape = tuple(shape)

        self._idxs = ravel_sparse_multi_index(multi_idx, self.shape)

        #TODO: if shape=() then data should be scalar array of value = fill_value
        if not np.iterable(data):
            data = tuple(itertools.repeat(data, self._idxs.shape[-1]))

        self._data = np.array(data, dtype=dtype, copy=copy)
        if self.data.ndim != 1:
            raise ValueError(f"'data' must be 1d, not {self.data.ndim}d")

        if self.data.shape[-1] != self.idxs.shape[-1]:
            raise ValueError("'data' does not have 1-1 correspondence with 'idxs'")

        if fill_value is None:
            fill_value = np.zeros((), self.dtype).item()
        self.fill_value = fill_value

        if sum_duplicates:
            self.sum_duplicates()
        else:
            self.sort_index()

    def __repr__(self):
        data = np.array2string(
            self.data,
            separator=", ",
            prefix=self._prefix_(0),
        )

        idxs = np.array2string(
            np.vstack(unravel_sparse_index(self.idxs, self.shape)),
            separator=", ",
            prefix=self._prefix_(1),
        )

        return self._repr_(
            data, idxs, self.shape, self.fill_value, self.dtype,
            ignore=('copy', 'sum_duplicates')
        )

    def __getitem__(self, idx):
        # TODO: slicing & fancy indexing & masks
        try:
            ravel = ravel_sparse_multi_index(idx, self.shape)
        except ValueError:
            raise IndexError('...')
            # IndexError: index 9 is out of bounds for axis 0 with size 4
            # IndexError: too many indices for array
        loc = self.idxs.searchsorted(ravel)
        if self.idxs[loc] == ravel:
            return self.data[loc]
        else:
            return self.fill_value

    def __eq__(self, other):
        if not is_broadcastable(self, other):
            msg = "elementwise comparison failed; this will raise an error in the future."
            warnings.warn(msg, DeprecationWarning)
            return False

        if isinstance(other, type(self)):
            # need to sum_duplicates since set routines used
            self.sum_duplicates()
            other.sum_duplicates()

            new_shape = ... # broadcast shape
            # may need to broadcast idxs and data arrays appropriately
            self_raveled = np.ravel_multi_index(self.idxs, new_shape)
            other_raveled = np.ravel_multi_index(other.idxs, new_shape)

            # eq must not have any xor, so rule that out first
            # and must have the same lengths
            if len(self_raveled) != len(other_raveled):
                return 'nevermind i want elemwise comparison'

            # elemwise compare eq with fill values incorporated
            self_and_other = np.intersect1d(self_raveled, other_raveled)
            # all false
            self_diff_other = np.setdiff1d(self_raveled, other_raveled)
            # all false
            other_diff_self = np.setdiff1d(other_raveled, self_raveled)

            new_raveled = np.hstack([self_and_other, self_diff_other, other_diff_self])
            new_idxs = np.unravel_index(new_raveled, new_shape)

            new_fill_value = self.fill_value + other.fill_value
            new_dtype = np.promote_types(self.dtype, other.dtype)
            new_data = np.empty(new_shape, new_dtype)

            # Oops this is for __add__
            # but I can generalize this idea in __array_func__ for __call__
            stop = self_and_other.size
            new_data[:stop] = self[self_and_other] + other[self_and_other]

            start = self_and_other.size
            stop += self_diff_other.size
            new_data[start:stop] = self[self_diff_other] + other.fill_value

            start += self_diff_other.size
            new_data[start:] = other[other_diff_self] + self.fill_value

            ...

        elif isinstance(other, np.ndarray):
            return np.array(self) == other
        else:
            mask = self.data == other
            return type(self)(mask, self.idxs, self.shape, fill_value=False, dtype=bool, sum_duplicates=False)

    def __array__(self):
        arr = np.full(self.shape, fill_value=self.fill_value, dtype=self.dtype)
        arr.ravel()[self._idxs] = self.data
        return arr

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            pass
        return NotImplemented

    def astype(self, dtype, casting='unsafe', copy=True):
        data = self._data.astype(dtype, casting=casting, copy=copy)
        unraveled = np.vstack(unravel_sparse_index(self.idxs, self.shape))
        if copy:
            return type(self)(data, unraveled, self.shape, self.fill_value, copy=copy)
        else:
            self._data = data
            return self

    def reshape(self, *shape, copy=True):
        shape = super().reshape(*shape)
        if not shape.size:
            # TODO: copy=copy
            return self

        unraveled = np.unravel_index(self._idxs, shape)

        return type(self)(
            self.data, unraveled, shape, self.fill_value, self.dtype, copy=copy
        )

    def sort_index(self):
        """Sort indices and and their respective data values.
        This is an in-place operation.
        """
        sorter = self.idxs.argsort()
        self._idxs = self.idxs[sorter]
        self._data = self.data[sorter]

    def sum_duplicates(self):
        """Eliminate duplicate entries by adding them together.
        This is an in-place operation.
        """
        if not self.size:
            return

        self._idxs, inverse = np.unique(self.idxs, return_inverse=True)
        data = np.zeros_like(self.idxs, dtype=self.dtype)
        np.add.at(data, inverse, self.data)
        self._data = data

    def transpose(self, *axes):
        ...

    #TODO: override from ABC
    @property
    def nbytes(self):
        return self.data.nbytes + self.idxs.nbytes

    #TODO: SparseArray.dtype.setter since self.idxs used
