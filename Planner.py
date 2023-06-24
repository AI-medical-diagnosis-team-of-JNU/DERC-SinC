import numpy as np
from collections import Counter
from preprocess import *

x = np.load("./Data/Train/x.npy")
x = preprocess(x)
y = np.loadtxt("./Data/Train/y.txt")


data_high = np.zeros([3692+1592, x.shape[1]])
label_high = np.zeros([3692+1592])

ind = 0

for i in range(y.shape[0]):
    if y[i] == 0 or y[i] == 5:
        data_high[ind] = x[i]
        label_high[ind] = y[i]
        ind += 1
assert ind == (3692+1592)
np.save("./Data/Train/Head/x.npy", data_high)
np.savetxt("./Data/Train/Head/y.txt", label_high)


data_high = np.zeros([779+509+694+503, x.shape[1]])
label_high = np.zeros([779+509+694+503])

ind = 0

for i in range(y.shape[0]):
    if y[i] == 2 or y[i] == 3 or y[i] == 4 or y[i] ==1:
        data_high[ind] = x[i]
        label_high[ind] = y[i]
        ind += 1
assert ind == (779+509+694+503)
np.save("./Data/Train/Medium/x.npy", data_high)
np.savetxt("./Data/Train/Medium/y.txt", label_high)


data_high = np.zeros([124+116+73+75, x.shape[1]])
label_high = np.zeros([124+116+73+75])

ind = 0

for i in range(y.shape[0]):
    if y[i] == 6 or y[i] == 7 or y[i] == 8 or y[i] ==9:
        data_high[ind] = x[i]
        label_high[ind] = y[i]
        ind += 1
assert ind == (124+116+73+75)
np.save("./Data/Train/Tail/x.npy", data_high)
np.savetxt("./Data/Train/Tail/y.txt", label_high)


y = y.astype(int)
result = np.bincount(y)
#result = Counter(y)
print(result)
print(y)
real_many_idx = np.argwhere(np.asarray(result)>500)
real_medium_idx = np.argwhere((np.asarray(result)>100)&(np.asarray(result)<=500))
real_few_idx = np.argwhere(np.asarray(result)<=100)
print(real_many_idx)
print(real_medium_idx)
print(real_few_idx)

