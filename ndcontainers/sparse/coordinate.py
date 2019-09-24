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
        self.idxs = np.atleast_2d(np.array(idxs, dtype=int, copy=copy))
        self.shape = shape
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
            ignore=('copy')
        )

    def __array__(self):
        arr = np.full(self.shape, fill_value=self.fill_value, dtype=self.dtype)
        arr[tuple(self.idxs.T)] = self.data
        return arr

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            data = None
            idxs = None
            shape = None
            fill_value = None
            dtype = None

            # may need to `flatten` shape first with change of `idxs`
            # to add coo arrays together

            # if add/sub constant -> change fill_value
            # else ...

            scalars = []
            for input in inputs:
                if isinstance(input, Number):
                    scalars.append(input)
                elif isinstance(input, type(self)):
                    scalars.append(input._i)
                    if shape is not None:
                        if shape != self.shape:
                            raise TypeError("inconsistent sizes")
                    else:
                        shape = self.shape
                else:
                    return NotImplemented

            return type(self)(ufunc(*scalars, **kwargs), idxs, shape, )
        else:
            return NotImplemented
