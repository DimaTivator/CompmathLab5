from abc import ABC, abstractmethod
import compmath._criterion as _criterion


class BasicSolver(ABC):
    """
    Basic class for all solvers

    Attributes
    -------------

    criterion: str, optional (default='abs_deviation') -- The stop criterion for the solver
    Possible values: 'abs_deviation', 'relative_diff', 'discrepancy_diff'

    eps: float, optional (default=1e-6) -- The error rate of the solver

    max_iter: int, optional (default=100) -- The maximum number of iterations until the method converges

    Methods
    -------------

    solve()

    """

    def __init__(
            self,
            criterion='abs_deviation',
            eps=1e-6,
            max_iter=100
    ):
        self.criterion = criterion
        self.eps = eps
        self.max_iter = max_iter

        # get criterion function by name
        try:
            self.crit_func = getattr(_criterion, self.criterion)
        except AttributeError:
            raise ValueError(f'Criterion function {self.criterion} not found')

    @abstractmethod
    def solve(self, **kwargs):
        pass
