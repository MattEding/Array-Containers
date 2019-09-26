import pytest
import numpy as np

from ndcontainers.sparse import SparseArrayABC


def test_abc_error():
    match = "Can't instantiate abstract class SparseArrayABC"
    with pytest.raises(TypeError, match=match):
        SparseArrayABC()
