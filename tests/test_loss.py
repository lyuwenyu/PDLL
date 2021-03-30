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

    def test_softmax(self, ):
        ''' '''
        data = torch.rand(2, 3, requires_grad=True)
        out = torch.softmax(data, 0)
        grad = torch.rand(2, 3)
        out.backward(grad)
        
        v = L.autograd.Tensor(data.data.numpy(), requires_grad=True)
        softmax = L.nn.Softmax(0)
        p = softmax(v)
        p.backward(grad.data.numpy())
        
        # np.testing.assert_almost_equal(p.data.numpy(), out.data.numpy(), decimal=4)
        # np.testing.assert_almost_equal(v.grad.numpy(), data.grad.numpy(), decimal=4)
        
        
    def test_crossentropy(self, ):
        '''
        '''
        
        loss = torch.nn.CrossEntropyLoss()
        logit = torch.randn(3, 5, requires_grad=True)
        target = torch.empty(3, dtype=torch.long).random_(5)
        output = loss(logit, target)
        output.backward(torch.zeros_like(output) + 0.6)

        v = L.autograd.Tensor(logit.data.numpy(), requires_grad=True)
        label = np.eye(5)[target.data.numpy()]

        mm = L.nn.CrossEntropyLoss()
        p = mm(v, label)
        p.backward(0.6)

        np.testing.assert_almost_equal(p.data.numpy(), output.data.numpy(), decimal=5)
        np.testing.assert_almost_equal(v.grad.numpy(), logit.grad.numpy(), decimal=5)
    


if __name__ == '__main__':
    
    unittest.main(verbosity=1)

