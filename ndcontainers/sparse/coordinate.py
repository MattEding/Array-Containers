from numbers import Number

import numpy as np

from .base import SparseArrayABC
from ..utils import NDArrayReprMixin


__all__ = ['CoordinateArray']

class CoordinateArray(
    np.lib.mixins.NDArrayOperatorsMixin, NDArrayReprMixin, SparseArrayABC
):
    def __init__(self, data, idxs, shape=None, fill_value=0, dtype=None, copy=False):
        self.data = np.array(data, dtype=dtype, copy=copy)
        if self.data.ndim != 1:
            raise ValueError(f"'data' must be 1d, not {self.data.ndim}d")

        self.idxs = np.atleast_2d(np.array(idxs, dtype=np.uint, copy=copy))
        if self.idxs.ndim != 2:
            raise ValueError(f"'idxs' must be 2d, not {self.idxs.ndim}d")

        if self.data.shape[-1] != self.idxs.shape[-1]:
            raise ValueError("'data' does not have 1-1 correspondence with 'idxs'")

        # sum data values that have duplicate indices
        self.idxs, inverse  = np.unique(self.idxs, axis=1, return_inverse=True)
        out = np.zeros_like(self.idxs[0], dtype=self.data.dtype)
        np.add.at(out, inverse, self.data)
        self.data = out

        # currently this does not allow for 0d/null shape --> sets shape to (1,)
        # the question is should a null CoordinateArray be allowed?
        min_shape = self.idxs.max(axis=1, initial=0) + 1
        if shape is None:
            self.shape = tuple(min_shape)
        else:
            shape = np.broadcast_arrays(shape).pop().astype(np.uint).ravel()
            if len(shape) != len(min_shape):
                raise ValueError(f"shape length does not match idxs length")
            elif np.any(shape < min_shape):
                raise ValueError(f"shape {tuple(shape)} values are too small for idxs")
            self.shape = tuple(shape)

        self._fill_value = None
        self.fill_value = fill_value 

    def __repr__(self):
        data = np.array2string(
            self.data,
            separator=", ",
            prefix=self._prefix_(0),
        )

        idxs = np.array2string(
            self.idxs,
            separator=", ",
            prefix=self._prefix_(1),
        )

        return self._repr_(
            data, idxs, self.shape, self.fill_value, self.dtype,
            ignore=('copy',)
        )

    def __array__(self):
        arr = np.full(self.shape, fill_value=self.fill_value, dtype=self.dtype)
        arr[tuple(self.idxs)] = self.data
        return arr

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        #TODO
        return NotImplemented

    def reshape(self, *shape, copy=True, order='C'):
        shape = np.array(shape).squeeze()
        neg = np.flatnonzero(shape < 0)
        if neg.size:
            if neg.size > 1:
                raise ValueError("can only specify one unknown dimension")
            shape[neg] = self.size // -np.prod(shape)

        if shape.prod() != self.size:
            raise ValueError(
                f"cannot reshape {type(self).__name__} of "
                f"size {self.size} into shape {tuple(shape)}"
            )

        raveled = np.ravel_multi_index(self.idxs, self.shape, order=order)
        unraveled = np.unravel_index(raveled, shape, order=order)
        idxs = np.array(unraveled, dtype=np.uint)
        return type(self)(
            self.data, idxs, shape, self.fill_value, self.dtype, copy=copy
        )

    @property
    def nbytes(self):
        return self.data.nbytes + self.idxs.nbytes
