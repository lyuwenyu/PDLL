import _init_path

import pdll 
import pdll.autograd
import pdll.nn
import pdll as L 

import torch
import numpy as np
import unittest

import os 

class Testing(unittest.TestCase):

    def test_eq(self, ):
        ''' 
        '''
        a = L.randn(1, 3)
        b = L.randn(1, 3)
        data = np.random.randn(1, 3)
        # print(a, data)

        # print(isinstance(a, L.Tensor), isinstance(b, L.Tensor))

        # c = a == b
        # d = a < b
        print(a == [1, 2, 3])




if __name__ == '__main__':
    
    unittest.main(verbosity=1)

