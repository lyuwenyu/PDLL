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

    def test_var(self, ):
        ''' '''
        data = torch.rand(2, 4, 3, 6, requires_grad=True)
        out = data.var(dim=(0, 2))
        out.mean().backward()

        v = L.autograd.Tensor(data.data.numpy(), requires_grad=True)
        o = v.var(axis=(0, 2))
        o.mean().backward()
        
        np.testing.assert_almost_equal(o.data.numpy(), out.data.numpy(), decimal=4)
        np.testing.assert_almost_equal(v.grad.numpy(), data.grad.numpy(), decimal=4)
        

        data = torch.rand(2, 4, 3, 6, requires_grad=True)
        out = data.var()
        out.mean().backward()

        v = L.autograd.Tensor(data.data.numpy(), requires_grad=True)
        o = v.var()
        
        o.mean().backward()
        
        np.testing.assert_almost_equal(o.data.numpy(), out.data.numpy(), decimal=4)
        np.testing.assert_almost_equal(v.grad.numpy(), data.grad.numpy(), decimal=4)
        

    def test_var_bias(self, ):
        ''' '''
        data = torch.rand(2, 4, 3, 6, requires_grad=True)
        out = data.var(dim=(0, 2), unbiased=False)
        out.mean().backward()

        v = L.autograd.Tensor(data.data.numpy(), requires_grad=True)
        o = v.var(axis=(0, 2), unbiased=False)
        o.mean().backward()
        
        np.testing.assert_almost_equal(o.data.numpy(), out.data.numpy(), decimal=4)
        np.testing.assert_almost_equal(v.grad.numpy(), data.grad.numpy(), decimal=4)
        

        data = torch.rand(2, 4, 3, 6, requires_grad=True)
        out = data.var(unbiased=False)
        out.mean().backward()

        v = L.autograd.Tensor(data.data.numpy(), requires_grad=True)
        o = v.var(unbiased=False)
        
        o.mean().backward()
        
        np.testing.assert_almost_equal(o.data.numpy(), out.data.numpy(), decimal=4)
        np.testing.assert_almost_equal(v.grad.numpy(), data.grad.numpy(), decimal=4)
      

if __name__ == '__main__':
    
    unittest.main(verbosity=1)

