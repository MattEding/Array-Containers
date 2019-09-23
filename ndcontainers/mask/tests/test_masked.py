import textwrap

import pytest
import numpy as np


from ndcontainers.mask import MaskedArray


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 3D Params
data_3d = np.array(
    [[[ 1, 3000, 9],
      [-1,    5, 4]],
        
     [[ 8,    0, 8],
      [ 8,    0, 8]]]
)

mask_3d = np.array(
    [[[1, 0, 0],
      [0, 1, 0]],
      
     [[0, 0, 0],
      [1, 1, 1]]]
)

ma_3d = MaskedArray(data_3d, mask_3d)

repr_3d = '''
MaskedArray(data=[[[  --, 3000,    9],
                   [  -1,   --,    4]],

                  [[   8,    0,    8],
                   [  --,   --,   --]]],
            mask=[[[ True, False, False],
                   [False,  True, False]],

                  [[False, False, False],
                   [ True,  True,  True]]],
            dtype=int64,
            fill_value=None)
'''.strip()

str_3d = '''
[[[  -- 3000    9]
  [  -1   --    4]]

 [[   8    0    8]
  [  --   --   --]]]
'''.strip()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Object Params
data_obj = np.array(
    [['a', 0, -1, 2., 3j],
     ['w', 4, -5, 6., 7j]],
    dtype=object
)

mask_obj = np.zeros_like(data_obj)
mask_obj[-1] = True

ma_obj = MaskedArray(data_obj, mask_obj)

repr_obj = '''
MaskedArray(data=[['a',   0,  -1, 2.0,  3j],
                  [ --,  --,  --,  --,  --]],
            mask=[[False, False, False, False, False],
                  [ True,  True,  True,  True,  True]],
            dtype=object,
            fill_value=None)
'''.strip()

str_obj = '''
[['a'   0  -1 2.0  3j]
 [ --  --  --  --  --]]
'''.strip()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Tests
@pytest.mark.parametrize(
    'masked_array, expected',
    [
        (ma_3d, repr_3d),
        (ma_obj, repr_obj),
    ]
)
def test_repr(masked_array, expected):
    assert repr(masked_array) == expected


@pytest.mark.parametrize(
    'masked_array, expected',
    [
        (ma_3d, str_3d),
        (ma_obj, str_obj),
    ]
)
def test_str(masked_array, expected):
    assert str(masked_array) == expected
