import numpy as np
import matplotlib.pyplot as plt
from random import random

def f(a, b, x):
    return a * x + b

x = np.linspace(0, 10, 100)
y_o = np.array([f(3, 5, t) + 5 * (-0.5 + random()) for t in x])
w = random()
b = 5
h = 0.01

for epoch in range(5):
    for i in range(x.shape[0]):
        y = f(w, b, x[i])
        dw = -h * (y - y_o[i]) * x[i]
        db = -h * (y - y_o[i])
        w = w + dw
        b = b + db

y_res = f(w, b, x)
plt.scatter(x, y_o)
plt.plot(x, y_res)
plt.show()
