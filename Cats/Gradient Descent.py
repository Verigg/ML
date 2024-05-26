import numpy as np

class GradientOptimizer:
    def __init__(self, oracle, x0):
        self.oracle = oracle
        self.x0 = x0

    def optimize(self, iterations, eps, alpha):
        x = self.x0.copy() # Copy the starting point
        for i in range(iterations):
            grad = self.oracle.get_grad(x) #Calculate the gradient for the point
            if np.linalg.norm(grad) < eps: #Comparing the result
                break
            x = x - alpha * grad #Finding the next step
        return x