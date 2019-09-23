import textwrap

import numpy as np

from ._constants import MASKED, NOMASK


class MaskedArray(np.lib.mixins.NDArrayOperatorsMixin):
    mask_display = '--'

    def __init__(self, data, mask=None, dtype=None, fill_value=None):
        self.data = np.array(data, dtype=dtype)
        self.mask = np.array(mask, dtype=bool)
        assert self.data.shape == self.mask.shape
        self.fill_value = fill_value

    def _formatter(self, x):
        if self.mask.ravel()[x]:
            fmt = self.mask_display
        else:
            val = self.data.ravel()[x]
            fmt = repr(val) if isinstance(val, str) else val
        return str(fmt).rjust(self._longest)

    def __repr__(self):
        longest = max(map(str, self.data[~self.mask]), key=len, default="")
        self._longest = max(len(longest), len(str(self.mask_display)))      

        name = type(self).__name__
        pad = " " * (len(name) + 1)  

        idxs = np.arange(self.data.size).reshape(self.data.shape)
        data = np.array2string(
            idxs,
            separator=", ",
            prefix=f"{name}(data=",
            formatter={'all': self._formatter}
        )

        mask = np.array2string(
            self.mask,
            separator=", ",
            prefix=f"{name}(mask="
        )

        text = """
        {name}(data={data},
        {pad}mask={mask},
        {pad}dtype={dtype},
        {pad}fill_value={fill_value})
        """

        text = textwrap.dedent(text).strip()
        return text.format(
            name=name,
            pad=pad,
            data=data,
            mask=mask,
            dtype=self.data.dtype,
            fill_value=self.fill_value,
        )

    def __str__(self):
        longest = max(map(str, self.data[~self.mask]), key=len)
        self._longest = max(len(longest), len(str(self.mask_display)))      

        idxs = np.arange(self.data.size).reshape(self.data.shape)
        data = np.array2string(
            idxs,
            formatter={'all': self._formatter}
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
