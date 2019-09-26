import itertools

import pytest
import numpy as np

from ndcontainers.sparse import CoordinateArray


@pytest.mark.parametrize(
    'coo_kw, arr_kw',
    [
        # identity matrix
        (
            # coo
            dict(
                data=[1, 1, 1],
                idxs=[
                        [0, 1, 2],
                        [0, 1, 2],
                ],
                shape=(3, 3),
            ),
            # arr
            dict(
                object=[
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ],
            ),
        ),
        # primes
        (
            #coo
            dict(
                data=[2, 3, 5, 7],
                idxs=[2, 3, 5, 7],
                shape=10
            ),
            # arr
            dict(
                object=[0, 0, 2, 3, 0, 5, 0, 7, 0, 0]
            ),
        ),
    ],
    ids=['identity_3x3', 'prime_10'],
)
def test_array_conversion(coo_kw, arr_kw):
    for dtype in (map(np.dtype, [int, float, complex, object])):
        coo = CoordinateArray(dtype=dtype, **coo_kw)
        arr = np.array(dtype=dtype, **arr_kw)
        dense = np.array(coo)
        try:
            assert np.allclose(dense, arr)
        except TypeError:
            assert np.all(dense == arr)



# @pytest.mark.skip()
# @pytest.mark.parametrized(
#     'shape',
#     [10, 24, ]
# )
# def test_reshape(shape):
#     data = []
#     idxs = [
#         [],
#         [],
#     ]
#     coo = CoordinateArray(data, idxs, shape)
#     arr = np.array(coo)



# def test_reshape_neg_error(shape):
#     pass


# def test_reshape_size_error(shape):
#     pass
