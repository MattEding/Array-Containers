import numpy as np

from ndcontainers.sparse import CoordinateArray


#    [[[0 0], [0 1], [0 2]],
#     [[1 0], [1 1], [1 2]]]
shape = (2, 3)

data1 = np.array([4, 8, 100])
idxs1 = np.array([[0, 0], [1, 1], [0, 1]])
coo1 = CoordinateArray(data1, idxs1, shape)

data2 = np.array([3, 5])
idxs2 = np.array([[0, 0], [0, 2]])
coo2 = CoordinateArray(data2, idxs2, shape)


rvl1 = np.ravel_multi_index(idxs1.T, shape)
rvl2 = np.ravel_multi_index(idxs2.T, shape)

idxs_flat = np.arange(np.prod(shape))
idxs_shaped = idxs_flat.reshape(shape)


# tpl1 = tuple(idxs1)
# tpl2 = tuple(idxs2)
# inter = np.intersect1d(tpl1, tpl2)
# xor = np.setxor1d(tpl1, tpl2)



arr_add = np.array([[  7, 100,   5],
                    [  0,   8,   0]])
# data_sum = [4, 8, 100]
# idx_sum = ...
# coo_sum = CoordinateArray()