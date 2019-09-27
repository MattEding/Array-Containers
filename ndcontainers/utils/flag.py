import numpy as np

from .descriptors import FlagDescriptor


__all__ = ['FlagsSynchronizer']

class FlagsSynchronizer:
    """Used to synchronize flags of multiple arrays.

    Parameters
    ----------
    flag_array : ndarray
        The main array to track state of.
        Notably for C vs F flags distinctions.

    arrays : ndarrays

    order: {'C', 'F'} (default='C')
    
    write : bool (default=None)
    
    align : bool (default=None)
    
    uic : bool (default=None)

    See Also
    --------
    np.ndarray.flags : information about flag attributes
    """
    aligned = FlagDescriptor()
    behaved = FlagDescriptor()
    c_contiguous = FlagDescriptor()
    carray = FlagDescriptor()
    contiguous = FlagDescriptor()
    f_contiguous = FlagDescriptor()
    farray = FlagDescriptor()
    fnc = FlagDescriptor()
    forc = FlagDescriptor()
    fortran = FlagDescriptor()
    num = FlagDescriptor()
    owndata = FlagDescriptor()
    writeable = FlagDescriptor()
    writebackifcopy = FlagDescriptor()

    __slots__ = ['_flag_array', '_array_refs']

    def __init__(self, flag_array, *arrays, order='C', write=None, align=None, uic=None):
        # Must use an array since making a flagsobj instance
        # from np.ctypes.flagsobj will not update values even
        # though it does not throw an error trying to do so.
        # 
        # Also shape (2, 2) is needed to have the 'order' parameter
        # trigger since vectors are both C and F order.
        # self._flag_array = np.empty((2, 2), dtype=bool, order=order)
        self._flag_array = flag_array
        self._array_refs = arrays
        self._flag_array.setflags(write, align, uic)

    def __repr__(self):
        return repr(self._flag_array.flags)

    def __getitem__(self, key):
        return self._flag_array.flags[key]
    
    def __setitem__(self, key, value):
        self._flag_array.flags[key] = value
        for arr in self._array_refs:
            arr.flags[key] = value
