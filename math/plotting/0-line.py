#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

# your code here
plt.plot( y, 'r')
plt.xticks(range(0, 11, 2))
plt.show()
