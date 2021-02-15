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

    def test_activation(self):

        a = np.random.rand(2, 3, 2) * 2 - 1

        v = L.autograd.Variable(a[...], requires_grad=True)
        v1 = L.nn.Tanh()(v)
        v1.mean().backward()

        t = torch.tensor(a, requires_grad=True)
        t1 = torch.nn.Tanh()(t)
        t1.mean().backward()

        np.testing.assert_almost_equal(v1.data, t1.data.numpy(), decimal=4)
        np.testing.assert_almost_equal(v.grad, t.grad.data.numpy(), decimal=4)


    def test_linear(self, ):
        data = np.random.rand(10, 10).astype(np.float32)

        v = L.autograd.Variable(data[...], requires_grad=True)
        t = torch.tensor(data, requires_grad=True)

        m_l = L.nn.Linear(10, 20)
        m_t = torch.nn.Linear(10, 20)

        m_l.weight.data = m_t.weight.data.numpy().transpose(1, 0)[...]
        m_l.bias.data = m_t.bias.data.numpy()[...]

        o_l = m_l(v)
        o_t = m_t(t)

        o_l.mean().backward()
        o_t.mean().backward()

        np.testing.assert_almost_equal(o_l.data, o_t.data.numpy(), decimal=5)
        np.testing.assert_almost_equal(m_l.weight.grad, m_t.weight.grad.numpy().transpose(1, 0), decimal=4)
        np.testing.assert_almost_equal(m_l.bias.grad, m_t.bias.grad.numpy(), decimal=4)
        np.testing.assert_almost_equal(v.grad, t.grad.numpy())


    def test_conv(self):

        data = np.random.rand(8, 16, 100, 100).astype(np.float32) * 2 - 1

        m_l = L.nn.Conv2d(16, 16, 5, 2, 1, dilation=3, groups=16)
        m_t = torch.nn.Conv2d(16, 16, 5, 2, 1, dilation=3, groups=16)

        v = L.autograd.Variable(data[...], requires_grad=True)
        t = torch.tensor(data, requires_grad=True)

        m_l.weight.data[...] = m_t.weight.data.numpy()[...]
        m_l.bias.data[...] = m_t.bias.data.numpy()[...]

        o_l = m_l(v)
        o_t = m_t(t)

        o_l.mean().backward()
        o_t.mean().backward()

        np.testing.assert_almost_equal(o_l.data, o_t.data.numpy(), decimal=5)
        np.testing.assert_almost_equal(m_l.weight.grad, m_t.weight.grad.numpy(), decimal=4)
        np.testing.assert_almost_equal(m_l.bias.grad, m_t.bias.grad.numpy(), decimal=4)
        np.testing.assert_almost_equal(v.grad, t.grad.numpy(), decimal=4)


    def test_max_pool(self):

        data = np.random.rand(8, 3, 100, 100).astype(np.float32)

        m_l = L.nn.Pool2d(3, 2, 1, mode='max')
        m_t = torch.nn.MaxPool2d(3, 2, 1, )

        v = L.autograd.Variable(data[...], requires_grad=True)
        t = torch.tensor(data, requires_grad=True)

        o_l = m_l(v)
        o_t = m_t(t)

        o_l.mean().backward()
        o_t.mean().backward()

        np.testing.assert_almost_equal(o_l.data, o_t.data.numpy(), decimal=5)
        np.testing.assert_almost_equal(v.grad, t.grad.numpy(), decimal=4)


    def test_avg_pool(self, ):

        data = np.random.rand(8, 3, 100, 100).astype(np.float32)

        m_l = L.nn.Pool2d(3, 2, 1, mode='avg')
        m_t = torch.nn.AvgPool2d(3, 2, 1, )

        v = L.autograd.Variable(data[...], requires_grad=True)
        t = torch.tensor(data, requires_grad=True)

        for _ in range(10):
            o_l = m_l(v)
            o_t = m_t(t)

            o_l.mean().backward()
            o_t.mean().backward()

            np.testing.assert_almost_equal(o_l.data, o_t.data.numpy(), decimal=5)
            np.testing.assert_almost_equal(v.grad, t.grad.numpy(), decimal=4)


    def test_bn(self, ):

        data = np.random.rand(8, 10, 100, 100).astype(np.float32)

        m_l = L.nn.BatchNorm2d(10)
        m_t = torch.nn.BatchNorm2d(10)

        v = L.autograd.Variable(data[...], requires_grad=True)
        t = torch.tensor(data, requires_grad=True)

        m_l.weight.data[...] = m_t.weight.data.numpy()[...]
        m_l.bias.data[...] = m_t.bias.data.numpy()[...]

        for _ in range(10):
            o_l = m_l(v)
            o_t = m_t(t)

            o_l.mean().backward()
            o_t.mean().backward()

            np.testing.assert_almost_equal(o_l.data, o_t.data.numpy(), decimal=4)
            np.testing.assert_almost_equal(m_l.weight.grad, m_t.weight.grad.numpy(), decimal=4)
            np.testing.assert_almost_equal(m_l.bias.grad, m_t.bias.grad.numpy(), decimal=4)
            np.testing.assert_almost_equal(v.grad, t.grad.numpy(), decimal=4)

            buffers = list(m_t.buffers())
            np.testing.assert_almost_equal(m_l.running_mean.data, buffers[0].data.numpy(), decimal=4)
            np.testing.assert_almost_equal(m_l.running_var.data, buffers[1].data.numpy(), decimal=4)


    def test_softmax(self, ):
        ''' '''
        data = torch.rand(2, 3, requires_grad=True)
        out = torch.softmax(data, 0)
        grad = torch.rand(2, 3)
        out.backward(grad)
        
        v = L.autograd.Variable(data.data.numpy(), requires_grad=True)
        softmax = L.nn.Softmax(0)
        p = softmax(v)
        p.backward(grad.data.numpy())
        
        np.testing.assert_almost_equal(p.data, out.data.numpy(), decimal=4)
        np.testing.assert_almost_equal(v.grad, data.grad.numpy(), decimal=4)
        


if __name__ == '__main__':
    
    unittest.main(verbosity=1)

