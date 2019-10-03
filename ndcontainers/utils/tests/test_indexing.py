import pytest
import numpy as np
from numpy import testing

from ndcontainers.utils import ravel_sparse_multi_index, unravel_sparse_index


# TODO: test () shape


@pytest.mark.parametrize(
    'shape',
    [(4, 4, 4), (10, 6, 30)],
    ids=repr,
)
def test_ravel_sparse_multi_index(shape):
    multi_index = np.array([
        [1, 0, 3, 0],
        [3, 0, 2, 3],
        [1, 0, 0, 2],
    ])
    sparse = ravel_sparse_multi_index(multi_index, shape)
    numpy = np.ravel_multi_index(multi_index, shape)
    testing.assert_array_equal(sparse, numpy)


@pytest.mark.parametrize(
    'shape',
    [(4, 4, 4), (10, 6, 30)],
    ids=repr,
)
def test_unravel_sparse_index(shape):
    indices = np.array([0, 4, 8, 10, 60, 39, 11, 48])
    sparse = np.vstack(unravel_sparse_index(indices, shape))
    numpy = np.vstack(np.unravel_index(indices, shape))
    testing.assert_array_equal(sparse, numpy)


def test_larger_dims():
    rng = np.random.RandomState(0)
    n = 32
    multi_index = np.arange(n * 10)
    rng.shuffle(multi_index)
    multi_index.shape = (n, -1)
    multi_index = multi_index % 4
    shape = shape = multi_index.max(axis=1) + 1

    match = 'too many dimensions passed to ravel_multi_index'
    with pytest.raises(ValueError, match=match):
        np.ravel_multi_index(multi_index, shape)

    ravel = ravel_sparse_multi_index(multi_index, shape)

    # match = ("dimensions are too large; arrays and shapes with a total size"
    #          " greater than 'intp' are not supported.")
    match = r"index \d+ is out of bounds for array with size -?\d+"
    with pytest.raises(ValueError, match=match):
        np.unravel_index(ravel, shape)

    unravel = unravel_sparse_index(ravel, shape)
    testing.assert_array_equal(np.vstack(unravel), multi_index)
