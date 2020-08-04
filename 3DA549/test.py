import numpy as np
from scipy.interpolate import interp1d

x = np.arange(29)
y = np.random.randint(0, 255, (29, 4, 4))
for i in range(4):
    for j in range(4):
        yy = y[:, i, j]
        li = interp1d(x, yy, kind='cubic')
        print(li(1.5))