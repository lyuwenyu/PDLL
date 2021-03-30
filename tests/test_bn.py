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


    def test_bn(self, ):

        data = np.random.rand(8, 10, 100, 100).astype(np.float32)

        m_l = L.nn.BatchNorm2d(10)
        m_t = torch.nn.BatchNorm2d(10)

        v = L.autograd.Tensor(data[...], requires_grad=True)
        t = torch.tensor(data, requires_grad=True)

        m_l.weight.storage = m_t.weight.data.numpy()[...]
        m_l.bias.storage = m_t.bias.data.numpy()[...]

        for _ in range(10):
            o_l = m_l(v)
            o_t = m_t(t)

            o_l.mean().backward()
            o_t.mean().backward()

            np.testing.assert_almost_equal(o_l.data.numpy(), o_t.data.numpy(), decimal=4)
            np.testing.assert_almost_equal(m_l.weight.grad.numpy(), m_t.weight.grad.numpy(), decimal=4)
            np.testing.assert_almost_equal(m_l.bias.grad.numpy(), m_t.bias.grad.numpy(), decimal=4)
            np.testing.assert_almost_equal(v.grad.numpy(), t.grad.numpy(), decimal=4)

            buffers = list(m_t.buffers())
            np.testing.assert_almost_equal(m_l.running_mean.data.numpy(), buffers[0].data.numpy(), decimal=4)
            np.testing.assert_almost_equal(m_l.running_var.data.numpy(), buffers[1].data.numpy(), decimal=4)


if __name__ == '__main__':
    
    unittest.main(verbosity=1)

