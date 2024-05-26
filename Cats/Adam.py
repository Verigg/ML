import numpy as np

class Adam:
    '''Represents an Adam optimizer

    Fields:
        eta: learning rate
        beta1: first moment decay rate
        beta2: second moment decay rate
        epsilon: smoothing term
    '''

    eta: float
    beta1: float
    beta2: float
    epsilon: float

    def __init__(self, *, eta: float = 0.1, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        '''Initalizes `eta`, `beta1` and `beta2` fields'''
        self.eta = eta
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

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
        m = np.zeros(x.size) # Starting first moment
        v = np.zeros(x.size) # Starting second moment
        for i in range(max_iter):
            grad = oracle.gradient(x) # Calculate the gradient for the point
            if np.linalg.norm(grad) < eps:
                break
            m = self.beta1 * m + (1.0 - self.beta1) * grad # Calculate first and second moment
            v = self.beta2 * v + (1.0 - self.beta2) * grad**2
            m_bias_corr = m / (1.0 - self.beta1 ** (i + 1)) # Calculate bias-corrected first and second moment
            v_bias_corr = v / (1.0 - self.beta2 ** (i + 1))
            x = x - self.eta * m_bias_corr / (np.sqrt(v_bias_corr) + self.epsilon) # Finding the next step
        return x