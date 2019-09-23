import inspect
import itertools

import numpy as np

from .constants import MASKED, NOMASK


__all__ = ['MaskedArray']

class MaskedArray(np.lib.mixins.NDArrayOperatorsMixin):
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

    def _formatter_factory(self): #FIXME  %timeit mask: 20ns  _vs_  %timeit ~mask: 46µs
        #: longest non-masked item        V this `~` is reason enough to _not_ have invert mask API
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
        sig = inspect.signature(type(self))
        params = list(sig.parameters)
        name = type(self).__name__

        idxs = np.arange(self.data.size).reshape(self.shape)
        data = np.array2string(
            idxs,
            separator=", ",
            prefix=f"{name}({params[0]}=",
            formatter={'all': formatter}
        )

        mask = np.array2string(
            self.mask,
            separator=", ",
            prefix=f"{name}({params[1]}="
        )

        head = params[0]
        width = len(name) + len(head) + 1
        tail = (p.rjust(width) for p in params[1:])
        values = (mask, self.dtype, self.fill_value)
        pair_seq = itertools.chain.from_iterable(zip(tail, values))
        formats = itertools.chain((name, head, data), pair_seq)
        text = "{}(" +  ",\n".join("{}={}" for p in params) + ")"
        return text.format(*formats)


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
        return type(self)(self.data[index], self.mask[index], self.fill_value)
        

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
    def ndim(self):
        return self.data.ndim
    
    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.shape
