""" A file to uphold the operations used in the numpy module for this assignment

Since there is assumed not to be any numpy module on the graders laptop this
    module will substitute as a replacement for those operations required.

I used numpy V1.9 to solve the algorithm initially and created this for
    compatability.

    -Matthew Carlis
"""

import copy
import math

class array(object):
    def __init__(self, items):
        if isinstance(items, list) and len(items) > 0 and isinstance(items[0], list):
            self.vector = [map(float, vals) for vals in items]
        elif isinstance(items, array):
            self.vector = copy.deepcopy(items)
        else:
            self.vector = [map(float, items)]
        self.T = self.vector

    def _make_transpose(self, items):
        transpose = []
        for r_cnt, row in enumerate(items):
            for v_cnt, val in enumerate(row):
                if r_cnt == 0:
                    transpose.append([val])
                else:
                    transpose[v_cnt].append(val)
        return transpose

    def _matrix_oper(self, other, func_obj, func_int):
        rv_mat = []
        if isinstance(other, array):
            for row in range(len(self.vector)):
                rv_mat.append([])
                for col in range(len(self.vector[row])):
                    try:
                        rv_mat[row].append(func_obj(row, col))
                    except Exception:
                        raise
        if isinstance(other, float) or isinstance(other, int):
            for row in range(len(self.vector)):
                rv_mat.append([])
                for col in range(len(self.vector[row])):
                    rv_mat[row].append(func_int(row, col))
        return array(rv_mat)

    def _primitive_oper(self, other, func_int):
        rv_mat = []
        if isinstance(other, float) or isinstance(other, int):
            for row in range(len(self.vector)):
                rv_mat.append([])
                for col in range(len(self.vector[row])):
                    rv_mat[row].append(func_int(row, col))
        return array(rv_mat)

    def __repr__(self):
        return self.vector.__repr__()
    def __len__(self):
        return len(self.vector)
    def __getitem__(self, index):
        return self.vector[index]
    def __div__(self, other):
        func_obj = lambda row, col: self.vector[row][col] / other.vector[row][col]
        func_int = lambda row, col: self.vector[row][col] / float(other)
        return self._matrix_oper(other, func_obj, func_int)
    def __rdiv__(self, other):
        func_int = lambda row, col: float(other) / self.vector[row][col]
        return self._primitive_oper(other, func_int)
    def __add__(self, other):
        func_obj = lambda row, col: self.vector[row][col] + other.vector[row][col]
        func_int = lambda row, col: self.vector[row][col] + float(other)
        return self._matrix_oper(other, func_obj, func_int)
    def __radd__(self, other):
        func_int = lambda row, col: self.vector[row][col] + float(other)
        return self._primitive_oper(other, func_int)
    def __sub__(self, other):
        func_obj = lambda row, col: self.vector[row][col] - other.vector[row][col]
        func_int = lambda row, col: self.vector[row][col] - float(other)
        return self._matrix_oper(other, func_obj, func_int)
    def __rsub__(self, other):
        func_int = lambda row, col: float(other) - self.vector[row][col]
        return self._primitive_oper(other, func_int)
    def __mul__(self, other):
        func_obj = lambda row, col: self.vector[row][col] * other.vector[row][col]
        func_int = lambda row, col: self.vector[row][col] * float(other)
        return self._matrix_oper(other, func_obj, func_int)
    def __rmul__(self, other):
        func_int = lambda row, col: self.vector[row][col] * float(other)
        return self._primitive_oper(other, func_int)

def full((rows, cols), value):
    matrix = []
    for cnt in range(rows):
        row = [float(value) for _x in range(cols)]
        matrix.append(row)
    return array(matrix)

def exp(value):
    if isinstance(value, array): # Hacky
        return math.exp(value.vector[0][0])
    return math.exp(float(value))

def dot(vect1, vect2):
    rv_sum = [0 for _x in range(len(vect1.vector))]
    for row in range(len(vect1.vector)):
        for col in range(len(vect1.vector[row])):
            rv_sum[row] += vect1.vector[row][col] * vect2.vector[row][col]
    rv = array(rv_sum)
    return rv

def matrix(vect):
    """ Return array(vect).  Do Nothing else.
    """
    return array(vect)

def sum(vect):
    """ Do Not Implement.
    """
    return vect
