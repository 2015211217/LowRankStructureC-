import numpy as np
from scipy.optimize import minimize
import math

np.set_printoptions(threshold=np.inf)
def cons1(a):
    return {'type': 'ineq', 'fun': lambda x: x[a] - math.pow(10, -10)}

def getNewWeight(INPUT_DIMENSION, eta, inputAccumulation, V, M, weightVector):
    fun = lambda x: np.ravel(eta * np.dot(np.transpose(x), inputAccumulation) + np.dot(
        np.dot(x, np.identity(INPUT_DIMENSION) + np.dot(np.dot(V.real, M), np.transpose(V.real))),
        np.transpose(x)))
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)
    number = []
    for i in range(INPUT_DIMENSION):
        number.append(i)
    a = tuple(map(cons1, number))
    cons = cons + a
    OptimizeResult = minimize(fun, weightVector, method='SLSQP', constraints=cons)
    weightVector = OptimizeResult['x']
    return weightVector
