from math import factorial


class Interpolation:
    def __init__(self, x, y):
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")

        self.x = x
        self.y = y
        self.n = len(x)
        self.diff_y = None

        self.build_difference_table()

    def build_difference_table(self):
        diff_y = [[0] * self.n for _ in range(self.n)]

        for i in range(self.n):
            diff_y[i][0] = self.y[i]

        for i in range(1, self.n):
            for j in range(self.n - i):
                diff_y[j][i] = diff_y[j + 1][i - 1] - diff_y[j][i - 1]

        self.diff_y = diff_y
        return

    def lagrange(self, val):
        sm = 0.0
        for i in range(self.n):
            term = 1.0
            for j in range(self.n):
                if j == i:
                    continue
                else:
                    term *= (val - self.x[j]) / (self.x[i] - self.x[j])
            sm += self.y[i] * term
        return sm

    def diff(self, k, i):
        if k == 0:
            return self.y[i]
        elif i + k >= self.n:
            raise ValueError("Index out of bounds")
        else:
            return (self.diff(k - 1, i + 1) - self.diff(k - 1, i)) / (self.x[i + k] - self.x[i])

    def newton(self, v):
        sm = self.y[0]
        for i in range(1, self.n):
            term = 1.0
            for j in range(i):
                term *= v - self.x[j]
            sm += self.diff(i, 0) * term
        return sm

    def gauss(self, v, h):
        a = len(self.y) // 2
        pn = None
        if v > self.x[a]:
            t = (v - self.x[a]) / h
            n = len(self.diff_y)
            pn = self.diff_y[a][0] + t * self.diff_y[a][1] + ((t * (t - 1)) / 2) * self.diff_y[a - 1][2]
            tn = t * (t - 1)
            for i in range(3, n):
                if i % 2 == 1:
                    n = int((i + 1) / 2)
                    tn *= (t + n - 1)
                    pn += ((tn / factorial(i)) * self.diff_y[a - n + 1][i])
                else:
                    n = int(i / 2)
                    tn *= (t - n)
                    pn += ((tn / factorial(i)) * self.diff_y[a - n][i])

        elif v < self.x[a]:
            t = (v - self.x[a]) / h
            n = len(self.diff_y)

            pn = self.diff_y[a][0] + t * self.diff_y[a - 1][1] + ((t * (t + 1)) / 2) * self.diff_y[a - 1][2]
            tn = t * (t + 1)
            for i in range(3, n):
                if i % 2 == 1:
                    n = int((i + 1) / 2)
                    tn *= (t + n - 1)
                else:
                    n = int(i / 2)
                    tn *= (t - n)

                fact = factorial(i)
                pn += (tn / fact) * self.diff_y[a - n][i]

        return pn
