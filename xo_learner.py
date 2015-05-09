""" Perceptron Algorithm.
-Matthew Carlis
"""
import sys
import copy
import numpy as np

class network(object):

    def __init__(self, matrix, trainset, epsilon=0.00001, initial_w=0.5, debug=False):
        self.epsilon = epsilon
        # Numpy Array.  A + B rows/col wise.
        #               A * B Row/Column wise.  Not Matrix Mult.
        #               5 + A Row/Column Wise.
        # np.array([1, 2, 3]) * np.array([4, 5, 6]) = np.array([4, 10, 18])
        self.matrix = np.array(matrix)
        self.trainset = np.array(trainset)
        self.weights = np.full((1, len(self.trainset[0])), 0.25)
        self.output_y = np.full((1, len(self.trainset)), 0.5)
        if debug:
            print 'matrix:\t\t', self.matrix.__repr__()
            print 'trainset:\t', self.trainset.__repr__()
            print 'weights:\t', self.weights.__repr__()
            print 'output_y:\t', self.output_y.__repr__()
            print 'output_y:\t', self.output_y[0].__repr__()
        self.run_train()

    def test_input(self):
        dot_product = np.dot(self.weights, np.array(np.matrix(self.matrix).T))
        ret_v = (1.0 / (1.0 + np.exp(-dot_product)))
        print ret_v,
        if ret_v > 0.4772 and ret_v < 0.5228:
            print "Tie X's & O's"
        elif ret_v >= 0.5228:
            print "MOSTLY O's"
        else:
            print "MOSTLY X's"


    def _matrix_logistic(self, train_indx):
        """ Return 1 / (1 + e ^ (- (W dot X)))
        """
        dot_product = np.dot(self.weights, np.array(np.matrix(self.trainset[train_indx]).T))
        return (1.0 / (1.0 + np.exp(-dot_product)))

    def _matrix_loss(self):
        """ for all x, y in Set. Sum((y - logistic(x))^2).
        """
        ret_sum = 0
        for cnt, val in enumerate(self.output_y[0]):
            ret_sum += np.sum((val - self._matrix_logistic(cnt))**2)
        return ret_sum

    def _alpha_t(self, time):
        return 1000.0 / (1000.0 + time)

    def _matrix_new_weights(self, h_xi, time, train_indx):
        """ new W_i function.  Matrix operation.
        """
        x_i, w_i, y_i = self.trainset[train_indx], self.weights, self.output_y[0][train_indx]
        product = self._alpha_t(time) * (y_i - h_xi) * h_xi * (1.0 - h_xi) * x_i
        return w_i + product

    def run_train(self):
        """ Run the training algorithm.
        """
        mod = len(self.trainset)
        start = 0
        for time in xrange(100000):
            train_indx = time % mod
            h_xi = self._matrix_logistic(train_indx)
            self.weights = self._matrix_new_weights(h_xi, time, train_indx)
            self.output_y[0][train_indx] = h_xi
            loss = self._matrix_loss() 
            if loss <= self.epsilon:
                print 'time:', time
                break
        if time == 100000 - 1:
            print 'Failed to Train in time.'
            return
        print 'loss:', loss
        self.test_input()


def get_matrix(file_name, matrix):
    """ Parse the input into a list of lists.
    """
    t_matrix = []
    ended = False
    with open(file_name, 'r') as file_d:
        for cnt, line in enumerate(file_d):
            row = []
            data = line.strip('\n')
            data = data.strip('\r')
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
        print '     ____________Failed Arguments____________'
        print 'USAGE: $python xo_learner.py trainers.txt input.txt'
        sys.exit(1)
    train_file, test_file = args[1], args[2]
    train_cases, test_matrix = [], []
    get_matrix(train_file, train_cases)
    get_matrix(test_file, test_matrix)
    return train_cases, test_matrix[0]

def build_vector_map(dataset, d_map):
    if isinstance(dataset, str):
        for val in dataset:
            if not d_map.has_key(val):
                if len(d_map) == 0:
                    d_map[val] = -1
                else:
                    d_map[val] = 1
            if len(d_map) == 2:
                break
        return len(d_map)
    if isinstance(dataset, list):
        for d_set in xrange(len(dataset)):
            size = build_vector_map(dataset[d_set], d_map)
            if size == 2:
                break
    return d_map

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



if __name__ == "__main__":
    ARGS = sys.argv
    # Map and Parse the input.
    if len(ARGS) == 3:
        TRAIN_SET, TEST_CASE = parse_input(ARGS)
    else:
        TRAIN_SET, TEST_CASE = parse_input(['', 'trainer.txt', 'test_file2.txt'])

    #D_MAP = build_vector_map(TRAIN_SET, {})
    D_MAP = {'X':-1, 'O':1}
    TRAIN_VALS, TEST_VALS = remap_values(TRAIN_SET, TEST_CASE, D_MAP)
    
    # Start the Training/Learning
    NET = network(TEST_VALS, TRAIN_VALS)
