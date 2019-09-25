from numbers import Number

import numpy as np

from .base import SparseArrayABC
from ..utils import NDArrayReprMixin


__all__ = ['CoordinateArray']

class CoordinateArray(
    np.lib.mixins.NDArrayOperatorsMixin, NDArrayReprMixin, SparseArrayABC
):
    def __init__(self, data, idxs, shape, fill_value=0, dtype=None, copy=False):
        self.data = np.array(data, dtype=dtype, copy=copy)
        self.idxs = np.atleast_2d(np.array(idxs, dtype=np.uint, copy=copy))
        self.shape = tuple(shape)
        self.fill_value = fill_value

        # assert (self.data.ndim == 1) or (not self.data.size)
        # assert (self.idxs.ndim == 2) or (not self.idxs.size)
        # if not self.data.size:
        #     assert not self.idxs.size
        # else:
        #     assert len(self.data) == len(self.idxs)
        #     assert self.idxs.shape[1] == self.ndim        
        # assert np.can_cast(self.fill_value, self.dtype)

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
        shape = np.array(shape)
        neg = np.flatnonzero(shape < 0)
        if neg.any():
            if len(neg) > 1:
                raise ValueError("can only specify one unknown dimension")
            shape[neg] = self.size // -np.prod(shape)

        if shape.prod() != self.size:
            raise ValueError(
                f"cannot reshape {type(self).__name__} of "
                f"size {self.size} into shape {shape}"
            )

        raveled = np.ravel_multi_index(self.idxs, self.shape, order=order)
        unraveled = np.unravel_index(raveled, shape, order=order)
        idxs = np.array(unraveled, dtype=np.uint)
        return type(self)(
            self.data, idxs, shape, self.fill_value, self.dtype, copy=copy
        )
