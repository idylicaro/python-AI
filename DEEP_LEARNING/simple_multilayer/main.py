# By: @Polaris000

import numpy as np
from multiLayer import MLP
from perceptron import Perceptron

train_data = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]])

target_xor = np.array(
    [
        [0],
        [1],
        [1],
        [0]])

target_nand = np.array(
    [
        [1],
        [1],
        [1],
        [0]])

target_or = np.array(
    [
        [0],
        [1],
        [1],
        [1]])

target_and = np.array(
    [
        [0],
        [0],
        [0],
        [1]])

#p_or = Perceptron(train_data, target_or)
#p_or.train()
#p_or.plot()

mlp = MLP(train_data, target_xor, 0.2, 5000)
mlp.train()
mlp.plot()
