from compmath.interpolation import *
import numpy as np
import matplotlib.pyplot as plt


x = [1, 2, 3, 4, 5]
y = [4, 8, 3, 9, 3]

i = Interpolation(x, y)

x_values = np.linspace(min(x) - 2, max(x) + 2, 100)

i.build_difference_table()
print(i.diff_y)

plt.scatter(x, y, label='Original', color='red')
# plt.plot(x_values, np.array([i.lagrange(x) for x in x_values]), label='Lagrange interpolation')
plt.plot(x_values, np.array([i.newton(x) for x in x_values]), label='Newton interpolation', color='green')
plt.plot(x_values, np.array([i.gauss(x, 1) for x in x_values]), label='Gauss interpolation', color='orange')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(min(y) - 4, max(y) + 4)
plt.xlim(min(x) - 2, max(x) + 2)
plt.grid(True)
plt.legend()
plt.show()


