import numpy as np
from matplotlib import pyplot as plt

def f(a, x):
    return a * x**2

def relu(w, x):
    return w * x if w * x >= 0 else 0

def drelu(w, x):
    return x if w * x >= 0 else 0

def y(w1, w2, w3, w4, x, shift_y=0):
    return w3 * relu(w1, x) + w4 * relu(w2, x) + shift_y

x = np.linspace(-2, 2, 15)
y_o = np.array([f(3, t) for t in x])
h = 0.001
w1, w2, w3, w4 = [-0.1, 0.1, 0.1, 0.1]
for epoch in range(1000):
    for i in range(x.shape[0]):
        dw4 = -h * (y(w1, w2, w3, w4, x[i]) - y_o[i]) * relu(w2, x[i])
        dw3 = -h * (y(w1, w2, w3, w4, x[i]) - y_o[i]) * relu(w1, x[i])
        dw2 = -h * (y(w1, w2, w3, w4, x[i]) - y_o[i]) * w4 * drelu(w2, x[i])
        dw1 = -h * (y(w1, w2, w3, w4, x[i]) - y_o[i]) * w3 * drelu(w1, x[i])
        w4 += dw4
        w3 += dw3
        w2 += dw2
        w1 += dw1

print(w1, w2, w3, w4)
shift_y = 3.5

y_res = np.array([y(w1, w2, w3, w4, t, shift_y) for t in x])
plt.scatter(x, y_o)
plt.plot(x, y_res)
plt.show()
