
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


from networkx import to_numpy_matrix
import networkx as nx

zkc = nx.karate_club_graph()
order = sorted(list(zkc.nodes()))

A = to_numpy_matrix(zkc, nodelist=order)
I = np.eye(zkc.number_of_nodes())

A_hat = A+I
D_hat = np.array(np.sum(A_hat, axis=0))[0]
D_hat = np.matrix(np.diag(D_hat))

W_1 = np.random.normal(
    loc=0, scale=1, size=(zkc.number_of_nodes(), 4))
W_2 = np.random.normal(
    loc=0, size=(W_1.shape[1], 2)
)

def gcn_layer(A_hat, D_hat, X, W):
    t = D_hat**-1 * A_hat * X * W
    return F.relu(torch.from_numpy(t)).numpy()

H_1 = gcn_layer(A_hat, D_hat, I, W_1)
H_2 = gcn_layer(A_hat, D_hat, H_1, W_2)

output = H_2
feature_repr = {
    node: np.array(output)[node] for node in zkc.nodes()
}

print(feature_repr)