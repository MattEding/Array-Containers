import numpy as np


def is_broadcastable(*arrays_or_shapes):
    """Determine whether arrays or shapes can be broadcast together.

    Parameters
    ----------
    arrays_or_shapes : array-like or tuple
        The arrays or shapes to check broadcasting rules of.

    Returns
    -------
    out : bool
        Boolean result of check whether the arrays or shapes are broadcastable.
    """
    shapes = (getattr(arr_shp, 'shape', arr_shp) for arr_shp in arrays_or_shapes)

    for vals in zip(*map(reversed, shapes)):
        mx = max(vals)
        if any(v not in (1, mx) for v in vals):
            return False
    return True


def get_broadcast_shape(*arrays_or_shapes):
    shapes = [getattr(arr_shp, 'shape', arr_shp) for arr_shp in arrays_or_shapes]
    shape = np.ones_like(max(shapes, key=len), dtype=int)

    for i, vals in enumerate(zip(*map(reversed, shapes)), start=1):
        mx = max(vals)
        if any(v not in (1, mx) for v in vals):
            raise ValueError("shapes cannot be broadcasted together")
        else:
            shape[-i] = mx # pylint: disable=unsupported-assignment-operation
    return tuple(shape)
