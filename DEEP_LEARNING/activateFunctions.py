# transfers function
import numpy as np

def step_function(sum_of):
    if sum_of >= 1:
        return 1
    else:
        return 0

# sigmoid muito usada para retornar probabilidades
def sigmoid_function(sum_of):
    return 1 / np.exp(-sum_of)

def tahn_function(sum_of):
    """ Hyperbolic tanget """
    return (np.exp(sum_of) - np.exp(-sum_of)) / (np.exp(sum_of) + np.exp(-sum_of))

def relu_function(sum_of):
    if sum_of >= 0:
        return sum_of
    return 0

def linear_function(sum_of):
    return sum_of

def softmax_function(sum_of:[numbers]):
    ex = np.exp(sum_of)
    return ex / ex.sum()

    print(sigmoid_function(2,1))
    print(tahn_function(2,1))
    print(relu_function(2,1))