import pytest
import numpy as np

from ndcontainers import abc
from ndcontainers.mask import MaskedArray
from ndcontainers.sparse import CoordinateArray


@pytest.mark.parametrize(
    'abstract_class',
    [abc.Array, abc.SparseArray]
)
def test_abc_error(abstract_class):
    match = "Can't instantiate abstract class"
    with pytest.raises(TypeError, match=match):
        abstract_class()


@pytest.mark.parametrize(
    'cls, bool',
    [
        (np.ndarray, True),
        (MaskedArray, True),
        (CoordinateArray, True),
        (list, False),
        (object, False),
    ]
)
def test_array_subclass(cls, bool):
    assert issubclass(cls, abc.Array) == bool


@pytest.mark.parametrize(
    'cls, bool',
    [
        (np.ndarray, False),
        (MaskedArray, False),
        (CoordinateArray, True),
        (list, False),
        (object, False),
    ]
)
def test_sparse_subclass(cls, bool):
    assert issubclass(cls, abc.SparseArray) == bool
