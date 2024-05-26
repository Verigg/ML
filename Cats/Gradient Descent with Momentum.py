import numpy as np


class GDM:
    '''Represents a Gradient Descent with Momentum optimizer

    Fields:
        eta: learning rate
        alpha: exponential decay factor
    '''

    eta: float # Moment damping rate
    alpha: float # Learning rate

    def __init__(self, *, alpha: float = 0.9, eta: float = 0.1):
        '''Initalizes `eta` and `alpha` fields'''
        self.alpha = alpha
        self.eta = eta

    def optimize(self, oracle: 'Oracle', x0: np.ndarray, *,
                 max_iter: int = 100, eps: float = 1e-5) -> np.ndarray:
        '''Optimizes a function specified as `oracle` starting from point `x0`.
        The optimizations stops when `max_iter` iterations were completed or
        the L2-norm of the gradient at current point is less than `eps`

        Args:
            oracle: function to optimize
            x0: point to start from
            max_iter: maximal number of iterations
            eps: threshold for L2-norm of gradient

        Returns:
            A point at which the optimization stopped
        '''
        x = x0 # Starting point
        v = np.zeros_like(x.size) # Starting velocity
        for i in range(max_iter):
            grad = oracle.gradient(x) # Calculate the gradient for the point
            if np.linalg.norm(grad) < eps: # Comparing the result
                break
            v = self.alpha * v + self.eta * grad # Calculate velocity
            x = x - v # Finding the next step
        return x