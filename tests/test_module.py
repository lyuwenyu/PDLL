

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

    def test_module(self, ):
        
        class MM(L.nn.Module):
            def __init__(self, ):
                super().__init__()
                self.conv = L.nn.Conv2d(10, 10, 3, 2, 1)
            
            def forward(self, ):
                pass


        class M(L.nn.Module):
            def __init__(self, ):
                super().__init__()

                self.conv = L.nn.Conv2d(10, 10, 3, 2, 1)
                self.l2 = L.nn.Linear(10, 10)
                self.conv1 = L.nn.Conv2d(10, 20, 3, 2, 1)
                self.mm = MM()

            def forward(self, ):
                pass

        m = M()
        print(m)


if __name__ == '__main__':
    
    unittest.main(verbosity=1)


