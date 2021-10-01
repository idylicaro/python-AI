from random import randint
from src.constants import *
from src.utils import *

actions = ['U', 'D', 'L', 'R']

actions_values = {
    "U": - MAX_COLUNS,
    "D": MAX_COLUNS,
    "L": -1,
    "R": 1,
}

linear_matrix = []  # [up,down,left,right,recompense]
initialize_vector(linear_matrix, TARGET_STATE, TARGET_R)

t = 0
while t < TIMES_EXECUTION:
    state = randint(0, TOTAL_STATES - 1)
    act = randint(0, 3)
    act_v = actions_values[actions[act]]
    print('state =', state, '-', 'act = ', act, '-', 'act_v', act_v)
    training(linear_matrix, state, act, act_v)
    t += 1

print("\ntraining is over\n")

# teste
i = 0
state = randint(0, TOTAL_STATES - 1)
while True:
    print('---step({0})---'.format(i))
    print('- state[{0}]'.format(state))
    if state == TARGET_STATE:
        print("*****************\nHas arrived!!!\n*****************")
        exit(0)
    print('-- Up ({0})'.format(linear_matrix[state][0]))
    print('-- Down ({0})'.format(linear_matrix[state][1]))
    print('-- Left ({0})'.format(linear_matrix[state][2]))
    print('-- Right ({0})'.format(linear_matrix[state][3]))
    print('-- R ({0})'.format(linear_matrix[state][4]))

    if linear_matrix[state][0] > linear_matrix[state][1] and linear_matrix[state][0] > linear_matrix[state][2] and linear_matrix[state][0] > linear_matrix[state][3]:
        state += actions_values['U']
    elif linear_matrix[state][1] > linear_matrix[state][0] and linear_matrix[state][1] > linear_matrix[state][2] and linear_matrix[state][1] > linear_matrix[state][3]:
        state += actions_values['D']
    elif linear_matrix[state][2] > linear_matrix[state][0] and linear_matrix[state][2] > linear_matrix[state][1] and linear_matrix[state][2] > linear_matrix[state][3]:
        state += actions_values['L']
    else:
        state += actions_values['R']

    print('Go to state -> {}'.format(state))
