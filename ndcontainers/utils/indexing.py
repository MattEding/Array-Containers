import operator
import itertools

import numpy as np


__all__ = [
    'ravel_sparse_multi_index',
    'unravel_sparse_index',
]


def _coefs(shape):
    iterable = itertools.chain([1], shape[1:][::-1])
    cum_prod = itertools.accumulate(iterable, operator.mul)
    return np.fromiter(cum_prod, dtype=np.uint)[::-1]


def ravel_sparse_multi_index(multi_index, dims):
    """Converts a tuple of index arrays into an array of flat indices.

    Parameters
    ----------
    multi_index : tuple of array_like
        A tuple of integer arrays, one array for each dimension.

    dims : tuple of ints
        The shape of array into which the indices from 'multi_index' apply.

    Returns
    -------
    raveled_indices : ndarray
        An array of indices into the flattened version of an array of
        dimensions 'dims'.

    Notes
    -----
    This can handle higher dimensions than 'np.ravel_multi_index', but when
    doing so, converting to a dense array will raise errors.
    """
    # TODO: multi_index as (tuple, list) 1d -> out.item()
    # TODO: error tests
    coefs = _coefs(dims)
    multi_index = np.asarray(multi_index).astype(np.uint)
    if multi_index.ndim == 1:
        multi_index = np.atleast_2d(multi_index).T

    try:
        min_shape = multi_index.max(axis=1) + 1
    except ValueError:
        min_shape = ()

    if np.any(dims < min_shape):
        raise ValueError("invalid entry in coordinates array")

    out = np.empty(multi_index.shape[1], dtype=np.uint)
    np.dot(multi_index.T, coefs, out=out)
    try:
        return out.item()
    except ValueError:
        return out


def unravel_sparse_index(indices, shape):
    """Converts a flat index or array of flat indices into a tuple of
    coordinate arrays.

    Parameters
    ----------
    indices : array_like
        An integer array whose elements are indices into the flattened version
        of an array of dimensions 'shape'.

    shape : tuple of ints
        The shape of the array to use for unraveling 'indices'.

    Notes
    -----
    This can handle higher dimensions than 'np.unravel_index', but when doing
    so, converting to a dense array will raise errors.
    """
    coefs = _coefs(shape)
    # not very efficient, may want to Cythonize this loop
    multi_index = []
    for modulo in coefs:
        multi, indices = divmod(indices, modulo)
        multi_index.append(multi)
    return tuple(multi_index)
