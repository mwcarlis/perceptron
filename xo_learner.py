""" Perceptron Algorithm.
-Matthew Carlis
"""

import sys



def get_matrix(file_name, matrix):
    """ Parse the input into a list of lists.
    """
    t_matrix = []
    ended = False
    with open(file_name, 'r') as file_d:
        for cnt, line in enumerate(file_d):
            row = []
            data = line.rstrip('\n')
            if len(data) == 0:
                matrix.append(t_matrix)
                t_matrix = []
                ended = True
                continue
            row = [_x for _x in data]
            t_matrix.append(row)
        if not ended:
            matrix.append(t_matrix)

def parse_input(args):
    if len(args) != 3:
        sys.exit(1)

    train_file = args[1]
    test_file  = args[2]
    train_cases = []
    test_matrix = []
    get_matrix(train_file, train_cases)
    get_matrix(test_file, test_matrix)
    return train_cases, test_matrix[0]

def build_vector_map(dataset, d_map):
    if isinstance(dataset, str):
        for val in dataset:
            if not d_map.has_key(val):
                d_map[val] = len(d_map)
            if len(d_map) == 2:
                break
        return len(d_map)
    if isinstance(dataset, list):
        for d_set in xrange(len(dataset)):
            size = build_vector_map(dataset[d_set], d_map)
            if size == 2:
                break
    return d_map


if __name__ == "__main__":
    ARGS = sys.argv
    TRAIN_SET, TEST_CASE = parse_input(ARGS)
    D_MAP = build_vector_map(TRAIN_SET, {})
    print D_MAP
    print TRAIN_SET
    print TEST_CASE
