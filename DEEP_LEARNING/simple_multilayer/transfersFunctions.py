import numpy as np


def step_function(sum_of):
    if sum_of >= 1:
        return 1
    else:
        return 0


# sigmoid muito usada para retornar probabilidades
def sigmoid_function(sum_of) -> float:
    return 1 / 1 + np.exp(-sum_of)


def sigmoid_derivative(x):
    return x * (1 - x)


def tahn_function(sum_of):
    """ Hyperbolic tanget """
    return (np.exp(sum_of) - np.exp(-sum_of)) / (np.exp(sum_of) + np.exp(-sum_of))


def relu_function(sum_of):
    if sum_of >= 0:
        return sum_of
    return 0


def linear_function(sum_of):
    return sum_of


def softmax_function(sum_of):
    ex = np.exp(sum_of)
    return ex / ex.sum()


def mean_absolute_error(x: [(int, float)]):
    result = 0
    for (expect, error) in x:
        result += abs(expect - error)
    return result / len(x)


def mean_square_error(x: [(int, float)]):
    result = 0
    for (expect, error) in x:
        result += (expect - error) ** 2
    return 1 / len(x) * result


def mean_root_error(x: [(int, float)]):
    result = 0
    for (expect, error) in x:
        result += (expect - error) ** 2
    return np.sqrt(1 / len(x) * result)
