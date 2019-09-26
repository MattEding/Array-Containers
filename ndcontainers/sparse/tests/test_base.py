import pytest
import numpy as np

from ndcontainers.sparse import SparseArray


def test_abc_error():
    match = "Can't instantiate abstract class"
    with pytest.raises(TypeError, match=match):
        SparseArray()
