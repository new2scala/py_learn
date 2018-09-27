
import numpy as np

A = np.matrix([
    [0, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [1, 0, 1, 0]
])

X = [[i, -i] for i in range(A.shape[0])]
X = np.matrix(X, dtype=float)

# print(X)

# now A*X => each node (row) represent a sum of neighbor features
#   gcn represent each node as an aggregate of its neighbors
# t = A*X
# print(t)


#t = A*X


A = A + np.matrix(np.eye(A.shape[0]))
D = np.sum(A, axis=0)
D = np.array(D)[0]
D = np.matrix(np.diag(D), dtype=float)
#print(D)

W = np.matrix([
    [1, -1],
    [-1, 1]
], dtype=float)
tA = D**-1*A*X*W

import torch.nn.functional as F
import torch

tA = F.relu(torch.from_numpy(tA))
print(tA)
# t = tA*X
# print(t)
