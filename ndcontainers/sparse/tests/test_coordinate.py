import itertools

import pytest
import numpy as np
from numpy import testing

from ndcontainers.sparse import CoordinateArray


#TODO: make these fixtures
# identity matrix
coo_ident_3x3 = CoordinateArray(
    data=[1, 1, 1],
    idxs=[
        [0, 1, 2],
        [0, 1, 2],
    ],
    shape=(3, 3),
    dtype=int,
)

arr_ident_3x3 = np.identity(3, dtype=int)

# primes
coo_prime_10 = CoordinateArray(
    data=[2, 3, 5, 7],
    idxs=[2, 3, 5, 7],
    shape=10,
)

arr_prime_10 = np.array([0, 0, 2, 3, 0, 5, 0, 7, 0, 0])


@pytest.mark.parametrize(
    'coo, arr',
    [(coo_ident_3x3, arr_ident_3x3), (coo_prime_10, arr_prime_10)],
    ids=['identity_3x3', 'prime_10'],
)
def test_array_conversion(coo, arr):
    #TODO extract types out so each outputs as own test
    for dtype in (map(np.dtype, [int, float, complex, object])):
        coo = coo.astype(dtype)
        arr = arr.astype(dtype)
        dense = np.array(coo)
        try:
            testing.assert_allclose(dense, arr)
        except TypeError:
            testing.assert_array_equal(dense, arr)


@pytest.fixture
def coo_reshape():
    data = [1, 2, 3]
    idxs = [0, 10, 20]
    shape = 24
    coo = CoordinateArray(data, idxs, shape)
    return coo


@pytest.mark.parametrize(
    'newshape',
    [(24, 1), (1, 24, 1), (2, 2, 2, 3), (8, 3), (-1, 4), (12, -5), (2, -1, 3)],
    ids=repr,
)
def test_reshape(newshape, coo_reshape):
    arr = np.array(coo_reshape)

    #TODO - factor out as meta parametrize
    for container in (tuple, list, np.array):
        newshape = container(newshape)

        coo_unpacked = coo_reshape.reshape(*newshape)
        coo_sequence = coo_reshape.reshape(newshape)
        coo_values = zip(coo_unpacked.__dict__.values(), coo_sequence.__dict__.values())
        for val1, val2 in coo_values:
            testing.assert_array_equal(val1, val2)

        arr_shaped = arr.reshape(newshape)

        testing.assert_array_equal(np.array(coo_unpacked), arr_shaped)
        testing.assert_array_equal(np.array(coo_sequence), arr_shaped)


@pytest.mark.parametrize(
    'shape',
    [(-1,), (-1, -1, 24), (2, -1, -1), (-3, -2, -4)],
    ids=repr,
)
def test_reshape_neg_error(shape, coo_reshape):
    pass


@pytest.mark.skip()
@pytest.mark.parametrize(
    'shape',
    [(), (-1, -1, 24), (2, -1, -1), (-3, -2, -4)],
    ids=repr,
)
def test_reshape_size_error(shape):
    pass


#TODO: to emulate array(0).reshape(())
