""" Perceptron Algorithm.
-Matthew Carlis
-Chris Nguyen
-Shunt Balushian
"""
import sys
import copy
import random

try:
    raise Exception('test')
    import numpy as t_np
    _TEST = t_np.full((1, 1), 1) # full only in latest numpy V1.9
    np = t_np
    print 'numpy as np'
except Exception:
    import my_numpy as np
    print 'my_numpy as np'

class Network(object):

    def __init__(self, matrix, trainset, epsilon=0.000000000000000000000000001, initial_w=0.5, debug=False):
        self.epsilon = epsilon
        # Numpy Array.  A + B rows/col wise.
        #               A * B Row/Column wise.  Not Matrix Mult.
        #               5 + A Row/Column Wise.
        # np.array([1, 2, 3]) * np.array([4, 5, 6]) = np.array([4, 10, 18])
        self.matrix = np.array(matrix)
        self.trainset = np.array([_x[:-1] for _x in trainset])
        self.weights = np.full((1, len(trainset[0][:-1])), initial_w)
        self.output_y = np.array([_x[-1:] for _x in trainset])
        if debug:
            print 'matrix:\t\t', self.matrix.__repr__()
            print 'trainset:\t', self.trainset.__repr__()
            print 'weights:\t', self.weights.__repr__()
            print 'output_y:\t', self.output_y.__repr__()

    def test_input(self):
        dot_product = (-1.0) * np.dot(self.weights, np.array(np.matrix(self.matrix).T))
        ret_v = (1.0 / (1.0 + np.exp(dot_product)))
        #if ret_v > 0.4772 and ret_v < 0.5228:
            #print "Tie X's & O's"
        #    return 'Tie'
        if ret_v >= 0.50:
            return "MOSTLY O's"
        else:
            return "MOSTLY X's"

    def _matrix_logistic(self, train_indx):
        """ Return 1 / (1 + e ^ (- (W dot X)))
        """
        dot_product = (-1.0) * np.dot(self.weights, np.array(np.matrix(self.trainset[train_indx]).T))
        return (1.0 / (1.0 + np.exp(dot_product)))

    def _matrix_loss(self):
        """ for all x, y in Set. Sum((y - logistic(x))^2).
        """
        ret_sum = 0.0
        for cnt, val in enumerate(self.output_y[0]):
            ret_sum += np.sum((val - self._matrix_logistic(cnt))**2)
        return ret_sum

    def _alpha_t(self, time):
        return 1000.0 / (1000.0 + time)

    def _matrix_new_weights(self, h_xi, time, train_indx):
        """ new W_i function.  Matrix operation.
        """
        try:
            x_i = np.array(self.trainset[train_indx]) 
            w_i = self.weights 
            y_i = self.output_y[train_indx][0]
            product = self._alpha_t(time) * (y_i - h_xi) * h_xi * (1.0 - h_xi) * x_i
        except:
            print self.output_y
            print self.output_y[0]
            print self.output_y[train_indx]
            print self.output_y[train_indx][0]
            print train_indx
            raise
        return w_i + product

    def run_train(self):
        """ Run the training algorithm.
        """
        mod = len(self.trainset)
        start = 0
        limit = 100000
        for time in xrange(limit):
            train_indx = time % mod
            h_xi = self._matrix_logistic(train_indx)
            self.weights = self._matrix_new_weights(h_xi, time, train_indx)
            self.output_y[train_indx][0] = h_xi
            loss = self._matrix_loss() 
            if loss <= self.epsilon:
                break
        if time == limit - 1:
            return 'Failed to Train in time.'
        return self.test_input()


def get_matrix(file_name, matrix):
    """
    """
    t_matrix = []
    ended = False
    file_d = open(file_name, 'r')
    data = file_d.read()
    data = data.split('\n')
    #print data
    row = []
    for item in data:
        item = item.strip('\r')
        if len(item) >= 3:
            for val in [_x for _x in item.strip('\r')]:
                row.append(val)
        elif len(item) == 1:
            row.append(item)
            matrix.append(row)
            row = []
        elif len(item) == 0:
            if len(row) != 0:
                matrix.append(row)
            row = []
            continue
    if len(row) != 0:
        matrix.append(row)
    return matrix

def parse_input(args):
    if len(args) != 3:
        print '     ____________Failed Arguments____________'
        print 'USAGE: $python xo_learner.py trainers.txt input.txt'
        sys.exit(1)
    train_file, test_file = args[1], args[2]
    train_cases, test_matrix = [], []
    get_matrix(train_file, train_cases)
    get_matrix(test_file, test_matrix)
    # print 'tc:', train_cases
    # print 'tm:', test_matrix
    return train_cases, test_matrix[0]

def remap_values(train_set, test_case, d_map):
    trainer = []
    for data_set in train_set:
        mat = []
        for d_row in data_set:
            for item in d_row:
                mat.append(d_map[item])
        trainer.append(mat)
    tester = []
    for data_set in test_case:
        for item in data_set:
            tester.append(d_map[item])
    return trainer, tester

def make_set(d_map, vect_cnt, val_cnt, train_set=False):
    data = ['X', 'O']
    if val_cnt % 2 == 0:
        print 'Invalid Dimensions/Size.  Odd numbers only.'
        sys.exit(1)
    ret_v = []
    for test in range(vect_cnt):
        x_cnt, o_cnt = 0, 0
        test_vals = []
        for cnt in range(val_cnt):
            value = random.choice(data)
            if 'X' in value:
                x_cnt += 1
            elif 'O' in value:
                o_cnt += 1
            test_vals.append(d_map[value])
        if train_set:
            if x_cnt > o_cnt:
                test_vals.append(d_map['X'])
            else:
                test_vals.append(d_map['O'])
        if vect_cnt > 1:
            ret_v.append(test_vals)
    if vect_cnt <= 1:
        return test_vals
    return ret_v

def random_test(train_set, d_map):
    import time
    data = ['X', 'O']
    in_size = 9
    test_samples = 9
    limit = 100
    average = 0
    print 'Input Size:', in_size
    for run in range(5):
        correct, incorrect, tie_cnt = 0, 0, 0
        init_t = time.time()
        #train_set = make_set(d_map, test_samples, in_size, True)
        #print len(train_set[0]), train_set[0]
        for test in range(limit):
            x_cnt, o_cnt = 0, 0
            test_vals = []
            for cnt in range(in_size):
                value = random.choice(data)
                if 'X' in value:
                    x_cnt += 1
                elif 'O' in value:
                    o_cnt += 1
                test_vals.append(d_map[value])
            if x_cnt == o_cnt:
                tie_cnt += 1
                continue
            net = Network(test_vals, train_set, debug=False)
            result = net.run_train()

            if o_cnt > x_cnt and "MOSTLY O's" in result:
                correct += 1
            elif o_cnt < x_cnt and "MOSTLY X's" in result:
                correct += 1
            else:
                incorrect += 1
        print 'time:', time.time() - init_t, 'Correct:', correct, 'incorrect:', incorrect, 'Ties:', tie_cnt
        average += correct
    print 'Average Correct:', float(average) / (100.0 * 5.0)


if __name__ == "__main__":
    ARGS = sys.argv
    # Map and Parse the input.
    D_MAP = {'X':-1, 'O':1}
    if len(ARGS) == 3:
        TRAIN_SET, TEST_CASE = parse_input(ARGS)
        TRAIN_VALS, TEST_VALS = remap_values(TRAIN_SET, TEST_CASE, D_MAP)
        NET = Network(TEST_VALS, TRAIN_VALS)
        print NET.run_train()
    else:
        TRAIN_SET, TEST_CASE = parse_input(['', 'trainer.txt', 'test_file.txt'])
        TRAIN_VALS, TEST_VALS = remap_values(TRAIN_SET, TEST_CASE, D_MAP)
        # for row in TRAIN_VALS:
        #     print 'len:', len(row), row
        # print 'len:', len(TEST_VALS), TEST_VALS
        random_test(TRAIN_VALS, D_MAP)
        # NET = Network(TEST_VALS, TRAIN_VALS, debug=True)
        #print NET.run_train()

