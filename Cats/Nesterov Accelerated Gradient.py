import numpy as np


class NesterovAG:
    '''Represents a Nesterov Accelerated Gradient optimizer

    Fields:
        eta: learning rate
        alpha: exponential decay factor
    '''

    eta: float # Learning rate
    alpha: float # Moment damping rate

    def __init__(self, *, alpha: float = 0.9, eta: float = 0.1):
        '''Initalizes `eta` and `aplha` fields'''
        self.alpha = alpha
        self.eta = eta

    def optimize(self, oracle: 'Oracle', x0: np.ndarray, *,
                 max_iter: int = 100, eps: float = 1e-5) -> np.ndarray:
        '''Optimizes a function specified as `oracle` starting from point `x0`.
        The optimizations stops when `max_iter` iterations were completed or
        the L2-norm of the current gradient is less than `eps`

        Args:
            oracle: function to optimize
            x0: point to start from
            max_iter: maximal number of iterations
            eps: threshold for L2-norm of gradient

        Returns:
            A point at which the optimization stopped
        '''
        x = x0 # Starting point
        v = np.zeros_like(x) # Starting velocity
        for i in range(max_iter):
            grad = oracle.gradient(x - self.alpha * v) # Calculate the gradient for the point
            if np.linalg.norm(grad, ord=2) < eps: # Comparing the result
                break
            v = self.alpha * v + self.eta * grad  # Calculate velocity (Nesterov accelerated gradient)
            x = x - v # Finding the next step
        return x