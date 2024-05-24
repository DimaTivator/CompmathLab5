from compmath.linalg import Matrix, get_diagonally_dominant
from compmath import _criterion
from dataclasses import dataclass

from ._base import BasicSolver


@dataclass
class SoleData:
    _n: int = 1
    _m: int = None
    _A: Matrix = None
    _b: Matrix = None

    def __post_init__(self):
        if self._m is None:
            self._m = self._n

        if self._A is not None:
            old_A = self.A.copy()
            self._A = Matrix([[0 for _ in range(self._m)] for _ in range(self._n)])

            self._A = Matrix([[old_A[i][j] if i < old_A.shape[0] and j < old_A.shape[1] else 0
                               for j in range(self._m)] for i in range(self._n)])
        else:
            self._A = Matrix([[0 for _ in range(self._m)] for _ in range(self._n)])

        if self._b is not None:
            old_b = self.b.copy()
            self._b = Matrix([[0] for _ in range(self._n)])

            self._b = Matrix([old_b[i] if i < old_b.shape[0] else [0] for i in range(self._n)])
        else:
            self._b = Matrix([[0] for _ in range(self._n)])

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        if value is not None:
            self._n = value
            self.__post_init__()

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, value):
        self._m = value
        self.__post_init__()

    @property
    def A(self):
        return self._A

    def set_A(self, i, j, value):
        if value is not None:
            self._A[i][j] = value

    @property
    def b(self):
        return self._b

    def set_b(self, i, value):
        if value is not None:
            self._b[i][0] = value


class SimpleIterationSolver(BasicSolver):

    def __init__(
            self,
            criterion='abs_deviation',
            eps=1e-6,
            max_iter=100
    ):
        super().__init__(criterion, eps, max_iter)

    def solve(self, **kwargs):
        """
        This method implements the Simple Iteration algorithm to solve the system of linear functions
        Ax = b

        Required keyword Arguments:
        - A: Matrix of shape (n, n)
        - b: Matrix of shape (n, 1)
        """

        A, b = kwargs['A'], kwargs['b']

        if A.det() == 0:
            raise ValueError('The matrix A is singular')

        A = get_diagonally_dominant(A)
        if A is None:
            raise ValueError("The matrix A is not diagonally dominant. Method can't be used")

        n = A.num_rows

        x = Matrix([[0] for _ in range(n)])

        C = Matrix([[-A[i][j] / A[i][i] if i != j else 0 for j in range(n)] for i in range(n)])
        b = Matrix([[b[i][0] / A[i][i]] for i in range(n)])

        res = [([val for val in x], '-')]

        for _ in range(self.max_iter):
            prev = x[:][0]
            x = C * x + b
            d = self.crit_func(x[:][0], prev)
            res.append(([val for val in x], d))
            if d < self.eps:
                break

        return res


def cramer_2d(coefficients, constants):
    coefficients_matrix = Matrix(coefficients)
    denominator = coefficients_matrix.det()

    if denominator == 0:
        return None

    x_matrix = Matrix([constants[:], [coefficients[1][0], coefficients[1][1]]])
    y_matrix = Matrix([[coefficients[0][0], coefficients[0][1]], constants[:]])

    x = x_matrix.det() / denominator
    y = y_matrix.det() / denominator

    return x, y
