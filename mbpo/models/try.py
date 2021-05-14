import numpy as np

a = np.array([0, 0, 1, 0])
b = np.array([1, 2, 3, 4])
b[a == 0] = 0
print(b)


np.random.randint()