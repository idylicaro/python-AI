from random import *

LEARN_RATING = 0.1

initial_weight1 = random()
initial_weight2 = random()
weight_bias = random()

input1 = int(input('Write the input (1)'))
input2 = int(input('Write the input (2)'))
bias = 1

error = 1

while error != 0:
    if input1 == 1 and input2 == 1:
        expected_result = 1
    else:
        expected_result = 0

    summation = initial_weight1 * input1
    summation += initial_weight2 * input2
    summation += weight_bias * bias

    if summation < 0:
        result = 0
    elif summation >= 0:
        result = 1

    print('-' * 100)
    print(f'Result ({result})')
    print('-' * 100)

    error = expected_result - result

    print(f'w1({initial_weight1})')
    print(f'w2({initial_weight2})')
    print(f'wb({weight_bias})')

    initial_weight1 = initial_weight1 + (LEARN_RATING * input1 * error)
    initial_weight2 = initial_weight2 + (LEARN_RATING * input2 * error)
    weight_bias = weight_bias + (LEARN_RATING * bias * error)

    print(f'error({error})')
