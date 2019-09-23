import inspect
import itertools

import numpy as np

from .constants import MASKED, NOMASK
from ..utils import NDArrayReprMixin


__all__ = ['MaskedArray']

class MaskedArray(np.lib.mixins.NDArrayOperatorsMixin, NDArrayReprMixin):
    """Masked array

    Parameters
    ----------
    data : array-like

    mask : array-like (default=None)

    dtype : ... (default=float)

    fill_value : ... (default=nan)

    Attributes
    ----------
    data
    dtype
    fill_value
    mask
    mask_display
    ndim
    shape
    size

    Methods
    -------
    reshape
    """
    mask_display = '--'

    def __init__(self, data, mask=None, dtype=float, fill_value=np.nan):
        self.data = np.array(data, dtype=dtype)

        if mask is None:
            self.mask = np.zeros_like(self.data, dtype=bool)
        else:
            self.mask = np.array(mask, dtype=bool)

        if self.data.shape != self.mask.shape:
            raise ValueError(
                "'data' and 'mask' do not match with shapes "
                f"{self.data.shape} {self.mask.shape}"
            )

        self.fill_value = fill_value

    def _formatter_factory(self):
        #: longest non-masked item
        longest = max(map(str, self.data[~self.mask]), key=len, default="")
        #: use mask_display if longer
        longest = max(len(longest), len(str(self.mask_display)))

        # can I pass *args, **kwargs / numpy.print_options here?
        def formatter(x):
            if self.mask.ravel()[x]:
                fmt = self.mask_display
            else:
                val = self.data.ravel()[x]
                fmt = repr(val) if isinstance(val, str) else val
            return str(fmt).rjust(longest)
        
        return formatter

    def __repr__(self):
        formatter = self._formatter_factory()

        data = np.array2string(
            np.arange(self.size).reshape(self.shape),
            separator=", ",
            prefix=self._prefix_(0),
            formatter={'all': formatter},
        )

        mask = np.array2string(
            self.mask,
            separator=", ",
            prefix=self._prefix_(1),
        )

        return self._repr_(data, mask, self.dtype, self.fill_value)

    def __str__(self):
        formatter = self._formatter_factory()   
        idxs = np.arange(self.data.size).reshape(self.data.shape)
        data = np.array2string(
            idxs,
            formatter={'all': formatter}
        )
        return data

    def __array__(self):
        type1 = np.result_type(self.fill_value)
        type2 = self.data.dtype
        dtype = np.promote_types(type1, type2)
        arr = self.data.astype(dtype)
        arr[self.mask] = self.fill_value
        return arr

    def __getitem__(self, index):
        if np.isscalar(self.data[index]):
            return self.data[index] if ~self.mask[index] else MASKED
        return type(self)(
            self.data[index], self.mask[index], self.dtype, self.fill_value
        )

    def __setitem__(self, index, item):
        if item is MASKED:
            self.mask[index] = True
        elif item is NOMASK:
            self.mask[index] = False
        else:
            if self.mask[index].any():
                raise IndexError
            self.data[index] = item

    def reshape(self, newshape):
        self.data = self.data.reshape()
        self.mask = self.mask.reshape()

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size
