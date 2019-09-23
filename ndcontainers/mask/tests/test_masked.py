import textwrap

import pytest
import numpy as np


from ndcontainers.mask import MaskedArray


# TODO # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# - float print options
# - default dtype
# - exceptions
# - use different mask_displays
# - do I want singleton mask value if nothing is masked?
# 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 1D Params
ma_1d_uint8 = MaskedArray(data=[0, 0], mask=[0, 1], dtype=np.uint8)

repr_1d_uint8 = '''
MaskedArray(data=[ 0, --],
            mask=[False,  True],
           dtype=uint8,
      fill_value=nan)
'''.strip()

str_1d_uint8 = '[ 0 --]'


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 3D Params
data_3d_int64 = np.array(
    [[[ 1, 3000, 9],
      [-1,    5, 4]],
        
     [[ 8,    0, 8],
      [ 8,    0, 8]]]
)

mask_3d_int64 = np.array(
    [[[1, 0, 0],
      [0, 1, 0]],
      
     [[0, 0, 0],
      [1, 1, 1]]]
)

ma_3d_int64 = MaskedArray(data_3d_int64, mask_3d_int64, dtype=int)

repr_3d_int64 = '''
MaskedArray(data=[[[  --, 3000,    9],
                   [  -1,   --,    4]],

                  [[   8,    0,    8],
                   [  --,   --,   --]]],
            mask=[[[ True, False, False],
                   [False,  True, False]],

                  [[False, False, False],
                   [ True,  True,  True]]],
           dtype=int64,
      fill_value=nan)
'''.strip()

str_3d_int64 = '''
[[[  -- 3000    9]
  [  -1   --    4]]

 [[   8    0    8]
  [  --   --   --]]]
'''.strip()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Object Params
data_2d_object = np.array(
    [['a', 0, -1, 2., 3j],
     ['w', 4, -5, 6., 7j]],
    dtype=object
)

mask_2d_object = np.zeros_like(data_2d_object)
mask_2d_object[-1] = True

ma_2d_object = MaskedArray(data_2d_object, mask_2d_object, dtype=object)

repr_2d_object = '''
MaskedArray(data=[['a',   0,  -1, 2.0,  3j],
                  [ --,  --,  --,  --,  --]],
            mask=[[False, False, False, False, False],
                  [ True,  True,  True,  True,  True]],
           dtype=object,
      fill_value=nan)
'''.strip()

str_2d_object = '''
[['a'   0  -1 2.0  3j]
 [ --  --  --  --  --]]
'''.strip()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Tests

ids = [f'ma_{ma.ndim}d_{ma.dtype}' for ma in (ma_1d_uint8, ma_2d_object, ma_3d_int64)]


@pytest.mark.parametrize(
    'masked_array, expected',
    [
        (ma_1d_uint8, repr_1d_uint8),
        (ma_2d_object, repr_2d_object),
        (ma_3d_int64, repr_3d_int64),
    ],
    ids=ids,
)
def test_repr(masked_array, expected):
    assert repr(masked_array) == expected


@pytest.mark.parametrize(
    'masked_array, expected',
    [
        (ma_1d_uint8, str_1d_uint8),
        (ma_2d_object, str_2d_object),
        (ma_3d_int64, str_3d_int64),
    ],
    ids=ids,
)
def test_str(masked_array, expected):
    assert str(masked_array) == expected
