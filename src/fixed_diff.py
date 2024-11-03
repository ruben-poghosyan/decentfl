import numpy as np
import matplotlib.pyplot as plt
mu = 0.5
def truncate(array):
    for i in range(len(array)):
        if array[i] <= 0:
            array[i] = 0.05
        if array[i] >= 1:
            array[i] = 0.95
a = list(map(lambda x: float(format(x, '.2f')), np.random.normal(mu, 0.1, 10000)))
truncate(a)
#b = [x - 0.2 for x in a]
b = []
for i in range(len(a)):
    if a[i] >= 0.5:
        b.append(0.05)
    else:
        b.append(0.95)
truncate(b)
final = []
for i in range(len(a)):
    final.append(a[i]/(a[i]+b[i]))
final = list(map(lambda x:float(format(x, '.2f')), final))
plt.axvline(0.5, color='red')
plt.hist(final,bins=200)
plt.show()