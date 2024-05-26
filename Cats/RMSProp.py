import numpy as np

class RMSProp:
    '''Represents an RMSProp optimizer

    Fields:
        eta: learning rate
        gamma: exponential decay factor
        epsilon: smoothing term
    '''

    eta: float
    gamma: float
    epsilon: float

    def __init__(self, *, eta: float = 0.1, gamma: float = 0.9, epsilon: float = 1e-8):
        '''Initalizes `eta`, `gamma` and `epsilon` fields'''
        self.eta = eta
        self.gamma = gamma
        self.epsilon = epsilon

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
        v = np.zeros(x.size) # Starting velocity
        for i in range(max_iter):
            grad = oracle.gradient(x) # Calculate the gradient for the point
            if np.linalg.norm(grad, ord=2) < eps:
                break
            v = self.gamma * v + (1.0 - self.gamma) * grad**2 # Calculate velocity (RMSprop)
            x = x - self.eta * grad / np.sqrt(v + self.epsilon) # Finding the next step
        return x