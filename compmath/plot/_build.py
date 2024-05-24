import matplotlib.pyplot as plt
import numpy as np


def plot_functions(functions, a, b, freq=1000):
    x_values = np.linspace(a, b, num=freq)
    fig, ax = plt.subplots()

    for func in functions:
        y_values = [func(x) for x in x_values]
        ax.plot(x_values, y_values, label=func.__name__)

    ax.grid()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axhline()

    return fig


def plot_equation_2d(functions, a, b, freq=500):
    x_values = np.linspace(a, b, num=freq)
    y_values = np.linspace(a, b, num=freq)
    fig, ax = plt.subplots()

    for func in functions:
        X, Y = np.meshgrid(x_values, y_values)
        f = np.vectorize(func)
        Z = f(X, Y)
        ax.contour(X, Y, Z, levels=[0])

    ax.grid()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axhline()

    return fig


def plot_function(func, x_min, x_max):
    x = np.linspace(x_min, x_max, 1000)
    y = func(x)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.grid(True)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    return fig
