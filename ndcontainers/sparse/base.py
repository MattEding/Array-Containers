import abc

import numpy as np


__all__ = ['SparseArray']


class SparseArray(abc.ABC):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # ABSTRACT

    # - - - - - - - - - - - - - - - - METHODS - - - - - - - - - - - - - - - -
    @abc.abstractmethod
    def astype(self, dtype, order='C', casting='unsafe', copy=True):
        """Copy of the array, cast to a specified type.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.

        #TODO: if I use Cython I will need to restrict the order accepted values
        order : {'C', 'F', 'A', 'K'}, (default='C')
            Controls the memory layout order of the result.
            'C' means C order, 'F' means Fortran order, 'A'
            means 'F' order if all the arrays are Fortran contiguous,
            'C' order otherwise, and 'K' means as close to the
            order the array elements appear in memory as possible.

        casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, (default='unsafe')
            Controls what kind of data casting may occur.

            * 'no' means the data types should not be cast at all.
            * 'equiv' means only byte-order changes are allowed.
            * 'safe' means only casts which can preserve values are allowed.
            * 'same_kind' means only safe casts or casts within a kind,
                like float64 to float32, are allowed.
            * 'unsafe' means any data conversions may be done.

        copy : bool, (default=True)
            By default, astype always returns a newly allocated array. If this
            is set to False, and the `dtype` and `order` requirements are
            satisfied, the input array is returned instead of a copy.

        Returns
        -------
        arr_t : ndarray
            Unless `copy` is False and the other conditions for returning the input
            array are satisfied (see description for `copy` input parameter), `arr_t`
            is a new array of the same shape as the input array, with dtype, order
            given by `dtype`, `order`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reshape(self, *shape, copy=True, order='C'):
        """Gives a new shape to a sparse array without changing its data.

        Parameters
        ----------
        shape : length-2 tuple of ints
            The new shape should be compatible with the original shape. If
            an integer, then the result will be a 1-D array of that length.
            One shape dimension can be -1. In this case, the value is
            inferred from the length of the array and remaining dimensions

        copy : bool, optional
            Indicates whether or not attributes of self should be copied
            whenever possible. The degree to which attributes are copied varies
            depending on the type of sparse array being used.

        order : {'C', 'F'}, optional
            Read the elements using this index order. 'C' means to read and
            write the elements using C-like index order; e.g. read entire first
            row, then second row, etc. 'F' means to read and write the elements
            using Fortran-like index order; e.g. read entire first column, then
            second column, etc.

        Returns
        -------
        reshaped_array : SparseArray
            A sparse array with the given `shape`, not necessarily of the same
            format as the current object.`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def setflags(write=None, align=None, uic=None):
        ...

    #FIXME: update docstring for SparseArrays
    @abc.abstractmethod
    def transpose(self, *axes):
        """Returns a view of the array with axes transposed.

        For a 1-D array this has no effect, as a transposed vector is simply the
        same vector. To convert a 1-D array into a 2D column vector, an additional
        dimension must be added. `np.atleast2d(a).T` achieves this, as does
        `a[:, np.newaxis]`.
        For a 2-D array, this is a standard matrix transpose.
        For an n-D array, if axes are given, their order indicates how the
        axes are permuted (see Examples). If axes are not provided and
        ``a.shape = (i[0], i[1], ... i[n-2], i[n-1])``, then
        ``a.transpose().shape = (i[n-1], i[n-2], ... i[1], i[0])``.

        Parameters
        ----------
        axes : None, tuple of ints, or `n` ints

        * None or no argument: reverses the order of the axes.

        * tuple of ints: `i` in the `j`-th place in the tuple means `a`'s
        `i`-th axis becomes `a.transpose()`'s `j`-th axis.

        * `n` ints: same as an n-tuple of the same ints (this form is
        intended simply as a "convenience" alternative to the tuple form)

        Returns
        -------
        out : ndarray
            View of `a`, with axes suitably permuted.

        See Also
        --------
        ndarray.T : Array property returning the array transposed.
        ndarray.reshape : Give a new shape to an array without changing its data.
        """
        raise NotImplementedError

    # - - - - - - - - - - - - - - - PROPERTIES - - - - - - - - - - - - - - -
    @property
    @abc.abstractmethod
    def nbytes(self):
        """Total bytes consumed by the elements of the sparse array."""
        raise NotImplementedError

    @nbytes.setter
    def nbytes(self, value):
        self._setter_not_writeable('nbytes')

    # - - - - - - - - - - - - - - - - DUNDERS - - - - - - - - - - - - - - - -
    @abc.abstractmethod
    def __array__(self):
        raise NotImplementedError 

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # PROPERTIES

    @property
    def data(self):
        """A read-only view of the underlying 'data' array."""
        view = self._data.view()
        view.setflags(write=False)
        return view

    @property
    def dtype(self):
        """Data-type of the sparse array’s elements."""
        return self.data.dtype

    @dtype.setter
    def dtype(self, value):
        if not np.can_cast(self.fill_value, value):
            raise ValueError(f"'fill_value' {self.fill_value!r} cannot be cast to {value!r}")
        dtype =  np.dtype(value)
        if dtype.itemsize != self.idxs.itemsize:
            raise ValueError(f"'data' {dtype!r} does not preserve shape with 'idxs'")
        self._data.dtype = dtype
        
    @property
    def fill_value(self):
        """The value of unspecified elements for this array."""
        return self._fill_value

    @fill_value.setter
    def fill_value(self, value):
        if not np.can_cast(value, self.dtype):
            raise ValueError(f"'fill_value' {value!r} cannot be cast to {self.dtype!r}")
        self._fill_value = value

    @property
    def flags(self):
        """Information about the memory layout of the sparse array.

        Attributes
        ----------
        C_CONTIGUOUS (C)
            The data is in a single, C-style contiguous segment.

        F_CONTIGUOUS (F)
            The data is in a single, Fortran-style contiguous segment.

        OWNDATA (O)
            The array owns the memory it uses or borrows it from another object.

        WRITEABLE (W)
            The data area can be written to.  Setting this to False locks
            the data, making it read-only.  A view (slice, etc.) inherits WRITEABLE
            from its base array at creation time, but a view of a writeable
            array may be subsequently locked while the base array remains writeable.
            (The opposite is not true, in that a view of a locked array may not
            be made writeable.  However, currently, locking a base object does not
            lock any views that already reference it, so under that circumstance it
            is possible to alter the contents of a locked array via a previously
            created writeable view onto it.)  Attempting to change a non-writeable
            array raises a RuntimeError exception.

        ALIGNED (A)
            The data and all elements are aligned appropriately for the hardware.

        WRITEBACKIFCOPY (X)
            This array is a copy of some other array. The C-API function
            PyArray_ResolveWritebackIfCopy must be called before deallocating
            to the base array will be updated with the contents of this array.

        FNC
            F_CONTIGUOUS and not C_CONTIGUOUS.

        FORC
            F_CONTIGUOUS or C_CONTIGUOUS (one-segment test).

        BEHAVED (B)
            ALIGNED and WRITEABLE.

        CARRAY (CA)
            BEHAVED and C_CONTIGUOUS.

        FARRAY (FA)
            BEHAVED and F_CONTIGUOUS and not C_CONTIGUOUS.

        Notes
        -----
        The `flags` object can be accessed dictionary-like (as in ``a.flags['WRITEABLE']``),
        or by using lowercased attribute names (as in ``a.flags.writeable``). Short flag
        names are only supported in dictionary access.

        Only the WRITEBACKIFCOPY, WRITEABLE, and ALIGNED flags can be
        changed by the user, via direct assignment to the attribute or dictionary
        entry, or by calling `ndarray.setflags`.

        The array flags cannot be set arbitrarily:

        - WRITEBACKIFCOPY can only be set ``False``.
        - ALIGNED can only be set ``True`` if the data is truly aligned.
        - WRITEABLE can only be set ``True`` if the array owns its own memory
        or the ultimate owner of the memory exposes a writeable buffer
        interface or is a string.

        Arrays can be both C-style and Fortran-style contiguous simultaneously.
        This is clear for 1-dimensional arrays, but can also be true for higher
        dimensional arrays.

        Even for contiguous arrays a stride for a given dimension
        ``arr.strides[dim]`` may be *arbitrary* if ``arr.shape[dim] == 1``
        or the array has no elements.
        It does *not* generally hold that ``self.strides[-1] == self.itemsize``
        for C-style contiguous arrays or ``self.strides[0] == self.itemsize`` for
        Fortran-style contiguous arrays is true.
        """
        return self._flags

    @flags.setter
    def flags(self, value):
        self._setter_not_writeable('flags')

    @property
    def idxs(self):
        """A read-only view of the underlying 'idxs' array."""
        view = self._idxs.view()
        view.setflags(write=False)
        return view

    @property
    def ndim(self):
        """The number of dimensions for this array."""
        return len(self.shape)
    
    @ndim.setter
    def ndim(self, value):
        self._setter_not_writeable('ndim')

    @property
    def nnz(self):
        """The number of "non-zero" elements."""
        return len(self.data)
    
    @nnz.setter
    def nnz(self, value):
        self._setter_not_writeable('nnz')

    @property
    def shape(self):
        """Tuple of sparse array dimensions."""
        return self._shape

    @shape.setter
    def shape(self):
        self._setter_not_writeable('shape')

    @property
    def size(self):
        """Number of elements in the sparse array including "non-zero" elements."""
        return np.prod(self.shape, dtype=int)

    @size.setter
    def size(self, value):
        self._setter_not_writeable('size')

    @property
    def sparsity(self):
        """TODO"""
        return self.nnz / self.size

    @sparsity.setter
    def sparsity(self):
        self._setter_not_writeable('sparsity')

    @property
    def T(self):
        return self.transpose()

    @T.setter
    def T(self):
        """The transposed sparse array."""
        self._setter_not_writeable('T')

    #TODO: change to a descriptor to also encompass deleters
    def _setter_not_writeable(self, name):
        raise AttributeError(f"attribute '{name}' of '{type(self).__name__}' objects is not writeable")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # DUNDERS

    def __bool__(self):
        if np.prod(self.shape, dtype=int) == 1:
            ...
        else:
            raise ValueError(
                "The truth value of a sparse array with more than one element is ambiguous. Use a.any() or a.all()"
            )

    def __len__(self):
        return self.shape[0]

    def __str__(self):
        return (
            f"<{type(self).__name__}: shape={self.shape}, nnz={self.nnz}, "
            f"dtype={self.dtype}, fill_value={self.fill_value}>"
        )
