import numpy as np

arr = np.arange(24)
arr1x24 = arr.reshape(1, 24)
arr2x12 = arr.reshape(2, 12)
arr3x8 = arr.reshape(3, 8)
arr4x6 = arr.reshape(4, 6)


idxs = np.array([[2, 3], [0, 0], [1, 1]])
raveled = np.ravel_multi_index(idxs.T, arr4x6.shape)
print(raveled)
'array([15,  0,  7])'

print(arr4x6[tuple(idxs.T)])
'array([15,  0,  7])' # this only works since the "indices" are equal to the "values"

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


arr1 = np.arange(10000).reshape(20, 10, 50)
arr2 = arr1.reshape(20, 500)

print(np.unravel_index(np.ravel_multi_index((10, 52), arr2.shape), arr1.shape))
print(np.unravel_index(np.ravel_multi_index((10, 1, 2), arr1.shape), arr2.shape))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Unravel Index

shape = (7, 6)
arr_flat = np.arange(np.prod(shape))
arr7x6 = arr_flat.reshape(shape)

value = 22
mask = (arr7x6 == value)

idx = np.unravel_index(value, shape)

mask[idx[0]] = 1
mask[:, idx[1]] = 1

arr7x6[~mask] = 0

print(idx)
# (3, 4)

print(arr7x6)
'''
        0  1  2  3 [4] 5
  +----------------------
 0 | [[ 0  0  0  0  4  0]
 1 |  [ 0  0  0  0 10  0]
 2 |  [ 0  0  0  0 16  0]
[3]|  [18 19 20 21 22 23]
 4 |  [ 0  0  0  0 28  0]
 5 |  [ 0  0  0  0 34  0]
 6 |  [ 0  0  0  0 40  0]]

'''