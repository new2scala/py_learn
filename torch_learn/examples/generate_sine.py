
import numpy as np
import torch

np.random.seed(2)

T = 20
Len = 1000
Num = 100

x = np.empty((Num, Len), dtype='int64')
b = np.array(range(Len))
#print(b)
ex = np.random.randint(-T, T, Num)
print(ex)
print(ex.reshape(Num, 1))
x[:] = b + ex.reshape(Num, 1)

data = np.sin(x*1.0/T).astype('float64')
torch.save(data, open('train.pt', 'wb'))
#print(x)