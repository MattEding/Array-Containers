#TODO: clean up docstrings since I have removed some parameters from
#       the ndarray versions (ie 'order' param)
import abc
import warnings

import numpy as np


__all__ = ['Array']

class Array(abc.ABC):
    @classmethod
    def __subclasshook__(cls, C):
        if cls is Array:
            if issubclass(C, np.ndarray):
                return True

            if any('__array__' in B.__dict__ for B in C.__mro__):
                return True
        return NotImplemented

    #TODO: change to a descriptor to also encompass deleters
    def _setter_not_writeable(self, name):
        raise AttributeError(
            f"attribute '{name}' of '{type(self).__name__}' objects is not writeable"
        )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # ABSTRACT

    # - - - - - - - - - - - - - - - - METHODS - - - - - - - - - - - - - - - -
    @abc.abstractmethod
    def astype(self, dtype, casting='unsafe', copy=True):
        """Copy of the array, cast to a specified type.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.

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
    def reshape(self, *shape, copy=True):
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

        Returns
        -------
        reshaped_array : SparseArray
            A sparse array with the given `shape`, not necessarily of the same
            format as the current object.`
        """
        if not len(shape):
            shape = np.array([])
        else:
            try:
                # `squeeze` for shape = dim1, ..., dimN *OR* (dim1, ..., dimN)
                shape = np.array(shape).astype(int, casting='safe').squeeze()
            except TypeError:
                raise ValueError("'shape' cannot consist of non-integers")

        is_neg = shape < 0
        neg_sum = is_neg.sum()
        if neg_sum > 1:
            raise ValueError("can only specify one unknown dimension")
        elif neg_sum:
            shape[is_neg] = self.size // np.prod(shape[~is_neg])

        if shape.prod() != self.size:
            raise ValueError(
                f"cannot reshape {type(self).__name__} of size"
                f" {self.size} into shape {tuple(shape)}"
            )
        return shape

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
    def dtype(self):
        """Data-type of the array’s elements."""
        raise NotImplementedError

    @dtype.setter
    @abc.abstractmethod
    def dtype(self, value):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def nbytes(self):
        """Total bytes consumed by the elements of the array."""
        raise NotImplementedError

    @nbytes.setter
    def nbytes(self, value):
        self._setter_not_writeable('nbytes')

    @property
    def ndim(self):
        """The number of dimensions for this array."""
        return len(self.shape)

    @ndim.setter
    def ndim(self, value):
        self._setter_not_writeable('ndim')

    @property
    @abc.abstractmethod
    def shape(self):
        """Tuple of sparse array dimensions."""
        raise NotImplementedError

    @shape.setter
    def shape(self, value):
        new = self.reshape(value, copy=False)
        self.__dict__.update(new.__dict__)

    @property
    def size(self):
        """Number of elements in the sparse array including "non-zero" elements."""
        return np.prod(self.shape, dtype=int)

    @size.setter
    def size(self, value):
        self._setter_not_writeable('size')

    @property
    def T(self):
        """The transposed sparse array."""
        return self.transpose()

    @T.setter
    def T(self):
        self._setter_not_writeable('T')

    # - - - - - - - - - - - - - - - - DUNDERS - - - - - - - - - - - - - - - -
    @abc.abstractmethod
    def __array__(self):
        raise NotImplementedError

    def __bool__(self):
        if not self.size:
            warnings.warn(
                "The truth value of an empty array is ambiguous. Returning False,"
                " but in future this will result in an error."
                " Use `array.size > 0` to check that an array is not empty.",
                DeprecationWarning
            )
        if np.prod(self.shape, dtype=int) != 1:
            raise ValueError(
                "The truth value of an array with more than one element is ambiguous."
                " Use a.any() or a.all()"
            )

    def __len__(self):
        return self.shape[0]
