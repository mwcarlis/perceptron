""" Perceptron Algorithm.
-Matthew Carlis
"""
import sys
import numpy as np

class network(object):

    def __init__(self, matrix, trainset, epsilon=0.0001, initial_w=0.5, debug=False):
        self.epsilon = epsilon
        # Numpy Array.  A + B rows/col wise.
        #               A * B Row/Column wise.  Not Matrix Mult.
        #               5 + A Row/Column Wise.
        # np.array([1, 2, 3]) * np.array([4, 5, 6]) = np.array([4, 10, 18])
        self.matrix = np.array(matrix)
        self.trainset = np.array(trainset)
        self.weights = np.full((1, len(self.trainset[0])), 0.5)
        self.output_y = np.full((1, len(self.trainset[0])), 0)
        if debug:
            print 'matrix:\t\t', self.matrix.__repr__()
            print 'trainset:\t', self.trainset.__repr__()
            print 'weights:\t', self.weights.__repr__()
            print 'output_y:\t', self.output_y.__repr__()

    def _matrix_logistic(self, train_indx):
        """ Return 1 / (1 + e ^ (- (W dot X)))
        """
        dot_product = np.dot(self.weights, np.array(np.matrix(self.trainset[train_indx]).T))
        return (1.0 / (1.0 + np.exp(-dot_product)))

    def _matrix_loss(self, train_indx):
        """ for all x, y in Set. Sum((y - logistic(x))^2).
        """
        return np.sum((self.output_y - self._matrix_logistic(train_indx))**2)

    def _alpha_t(self, time):
        return 1000.0 / (1000.0 + time)

    def _matrix_new_weights(self, time, train_indx):
        """ new W_i function.  Matrix operation.
        """
        x_i, w_i, y_i = self.trainset[train_indx], self.weights, self.output_y
        h_xi = self._matrix_logistic(train_indx)
        product = self._alpha_t(time) * (y_i - h_xi) * h_xi * (1.0 - h_xi) * x_i
        return w_i + product

    def run_train(self):
        """ Run the training algorithm.
        """
        mod = len(self.trainset)
        start = 0
        for time in xrange(100000):
            train_indx = time % mod
            self.weights = self._matrix_new_weights(time, train_indx)
            loss = self._matrix_loss(train_indx) 
            if loss <= self.epsilon:
                break
        print '\ntrainset:', self.trainset.__repr__()
        print 'weights:', self.weights.__repr__()
        print 'loss:', loss
        print 'time:', time
        print 'Done Running'


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
        print 'Failed Arguments'
        return
        #sys.exit(1)
    train_file, test_file = args[1], args[2]
    train_cases, test_matrix = [], []
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

def remap_values(train_set, test_case, d_map):
    trainer = []
    for data_set in train_set:
        mat = []
        for d_row in data_set:
        #    row = []
            for item in d_row:
                mat.append(d_map[item])
        #        row.append(d_map[item])
        #    mat.append(row)
        trainer.append(mat)
    tester = []
    for data_set in test_case:
        #row = []
        for item in data_set:
            #row.append(d_map[item])
            tester.append(d_map[item])
        #tester.append(row)
    return trainer, tester



if __name__ == "__main__":
    ARGS = sys.argv
    # Map and Parse the input.
    #TRAIN_SET, TEST_CASE = parse_input(ARGS)
    TRAIN_SET, TEST_CASE = parse_input(['', 'trainer.txt', 'test_file.txt'])
    #TRAIN_SET, TEST_CASE = 
    D_MAP = build_vector_map(TRAIN_SET, {})
    TRAIN_VALS, TEST_VALS = remap_values(TRAIN_SET, TEST_CASE, D_MAP)
    
    # Start the Training/Learning
    NET = network(TEST_VALS, TRAIN_VALS)
