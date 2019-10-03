# N-dimensional Array Containers
NumPy compatible array-like structures.

## Sparse Arrays
__Goal:__ Replace `scipy.sparse` since it only handles 2d matrices.

### CoordinateArray

```
>>> from ndcontainers.sparse import CoordinateArray
>>> shape = (10, 10)
>>> xs  = np.array([0, 0, 1, 3, 1, 0, 0])
>>> ys = np.array([0, 2, 1, 3, 1, 0, 0])
>>> idxs = np.vstack([xs, ys])
>>> data = np.arange(7) + 1
>>> coo = CoordinateArray(data, idxs, shape)
```

Nice `repr` and `str` formatting
```
>>> coo
CoordinateArray(data=[14,  2,  8,  4],
                idxs=[[0, 0, 1, 3],
                      [0, 2, 1, 3]],
               shape=(10, 10),
          fill_value=0,
               dtype=int64)
>>> print(coo)
<CoordinateArray: shape=(10, 10), nnz=4, dtype=int64, fill_value=0>
```

Efficient internal index storage by raveling multi-index into a flattened array
and combining duplicate indices.
```
>>> coo.idxs
array([ 0,  2, 11, 33], dtype=uint64)
>>> np.unravel_index(coo.idxs, coo.shape)
(array([0, 0, 1, 3]), array([0, 2, 1, 3]))
>>> coo.nbytes
64
>>> idxs.nbytes + data.nbytes
168
```

Basic indexing support (advanced indexing in the works)
```
>>> coo[0, 2]
2
>>> coo[1, 7]
0
```

Conversion to `np.ndarray`
```
>>> import numpy as np
>>> arr = np.array(coo)
>>> arr
array([[14,  0,  2,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  8,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  4,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])
>>> arr.nbytes
800
```

### Work in progress
- fancy indexing
- ufuncs
- etc.

## Masked Array
__Goal:__ better performance than `numpy.ma` by using Cython, and to not have side-effects when doing computing.
```
>>> from ndcontainers.mask import MaskedArray
>>> from numpy import ma
>>> data = np.array(
    [[[ 1, 3000, 9],
      [-1,    5, 4]],

     [[ 8,    0, 8],
      [ 8,    0, 8]]]
)
>>> mask = np.array(
    [[[1, 0, 0],
      [0, 1, 0]],

     [[0, 0, 0],
      [1, 1, 1]]]
)
```

Cleaner string outputs
```
>>> ndc_mask = MaskedArray(data, mask, dtype=int)
>>> print(ndc_mask)
[[[  -- 3000    9]
  [  -1   --    4]]

 [[   8    0    8]
  [  --   --   --]]]
>>> np_mask = ma.array(data, mask=mask)
>>> print(np_mask)
[[[-- 3000 9]
  [-1 -- 4]]

 [[8 0 8]
  [-- -- --]]]
```

### Work in progress
- a lot


## Jagged Arrays
__Goal__: Implement [jagged arrays](https://en.wikipedia.org/wiki/Jagged_array) without using `dtype=object`

```
>>> import numpy as np
>>> data = [[0, 1], [2], [3, 4, 5]]
>>> jag = np.array(data)
>>> jag
array([list([0, 1]), list([2]), list([3, 4, 5])], dtype=object)
```

### Work in progress
- *have not started*

```
>>> # want the following; not yet implemented
>>> from ndcontainers.jagged import JaggedArray
>>> jag = JaggedArray(data)
>>> jag
JaggedArray(data=[[0, 1],
                  [2],
                  [3, 4, 5]],
           dtype=int64)
>>> jag.sum()
15
>>> jag.sum(axis=0)
array([ 1,  2, 12])
>>> jag.sum(axis=1)
array([5, 5, 5])