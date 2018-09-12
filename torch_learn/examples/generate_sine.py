
import numpy as np
import torch

np.random.seed(2)

Max = 20
Len = 1000
Num = 100

x = np.empty((Num, Len), dtype='int64')
b = np.array(range(Len))
#print(b)
ex = np.random.randint(-Max, Max, Num)
print('%d samples in [%d, %d]'%(Num, -Max, Max))

reshaped = ex.reshape(Num, 1)
print('reshaped to \n{}'.format(reshaped))
x[:] = b + reshaped
#print('after plus {}\n{}'.format(b, x))

data = np.sin(x*1.0/Max).astype('float64')
torch.save(data, open('train.pt', 'wb'))
#print(x)