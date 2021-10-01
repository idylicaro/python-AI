from src.constants import LEARNING_RATE, TOTAL_STATES, TARGET_STATE


def initialize_vector(vector, target_state, recompense_value):
    for i in range(0, TOTAL_STATES, 1):
        vector.append([0, 0, 0, 0, 0])
    vector[target_state][4] = recompense_value


def training(vector, state, action_pos, action_value):
    if state + action_value >= 0 and state + action_value < TOTAL_STATES:
        evaluate_recompense(vector, state, action_pos, action_value)


def evaluate_recompense(vector, state, action_pos, action_value):
    vector[state][action_pos] = vector[state + action_value][4] + (LEARNING_RATE * max(
        vector[state + action_value][0],
        vector[state + action_value][1],
        vector[state + action_value][2],
        vector[state + action_value][3]
    ))
