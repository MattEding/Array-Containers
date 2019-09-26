import numpy as np

from .descriptors import FlagDescriptor


__all__ = ['FlagsObj']

class FlagsObj:
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
    updateifcopy = FlagDescriptor()
    writeable = FlagDescriptor()
    writebackifcopy = FlagDescriptor()

    __slots__ = ['_flag_array', '_array_refs']

    def __init__(self, *arrays, write=None, align=None, uic=None):
        # must use an array since making a flagsobj instance
        # from np.ctypes.flagsobj will not update values even
        # though it does not throw an error trying to do so
        self._flag_array = np.array(None)
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

o = np.ones(1)
z = np.zeros(1)
f = FlagsObj(o, z)
