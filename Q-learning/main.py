from random import randint
from .utils import *

## CONSTANTS ##

TIMES_EXECUTION = 10000

LEARNING_RATE = 0.9
MAX_COLUNS = 6
MAX_ROWS = 6

TOTAL_STATES = MAX_COLUNS * MAX_ROWS

TARGET_STATE = 36  # STATE THAT HAS RECOMPENSE
TARGET_R = 10

actions = ['U', 'D', 'L', 'R']

actions_values = {
    "U": - MAX_COLUNS,
    "D": MAX_COLUNS,
    "L": -1,
    "R": 1,
}

linear_matrix = [TOTAL_STATES]  # [up,down,left,right,recompense]
initialize_vector(linear_matrix, TARGET_STATE, TARGET_R)


t = 0
while t < TIMES_EXECUTION:
    # algorithm
    print(randint(0, 3))
    t += 1


# r + LEARNING_RATE + MAX()



