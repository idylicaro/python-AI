def initialize_vector(vector, target_state, recompense_value):
    for x in vector:
        x = [0, 0, 0, 0, 0]

    vector[target_state][4] = recompense_value
