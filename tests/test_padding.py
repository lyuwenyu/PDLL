
import _init_path
import unittest

import numpy as np
import torch
import pdll as L

class Testing(unittest.TestCase):

    def test_padding_functional(self, ):
        ''' '''
        for padding in [2, [2, 3], [2, 3, 4, 5], ]:
            t = torch.rand(4, 5, 6, 7, requires_grad=True)
            ot = torch.nn.ZeroPad2d(padding=padding)(t)
            v = L.from_numpy(t.data.numpy()[...], requires_grad=True)
            ov = L.nn.functional.zero_pad2d(v, padding)
            ot.mean().backward()
            ov.mean().backward()

            np.testing.assert_almost_equal(ov.data.numpy(), ot.data.numpy())
            np.testing.assert_almost_equal(v.grad.numpy(), t.grad.data.numpy())


    def test_zeropadding_module(self, ):
        ''' '''
        for padding in [2, [2, 3], [2, 3, 4, 5], ]:
            t = torch.rand(4, 5, 6, 7, requires_grad=True)
            v = L.from_numpy(t.data.numpy()[...], requires_grad=True)

            ot = torch.nn.ZeroPad2d(padding=padding)(t)
            ov = L.nn.ZeroPad2d(padding=padding)(v)
            ot.mean().backward()
            ov.mean().backward()

            np.testing.assert_almost_equal(ov.data.numpy(), ot.data.numpy())
            np.testing.assert_almost_equal(v.grad.numpy(), t.grad.data.numpy())


    def test_constantpadding_module(self, ):
        ''' '''
        for padding in [2, [2, 3], [2, 3, 4, 5], ]:
            t = torch.rand(4, 5, 6, 7, requires_grad=True)
            v = L.from_numpy(t.data.numpy()[...], requires_grad=True)

            ot = torch.nn.ConstantPad2d(padding=padding, value=2.)(t)
            ov = L.nn.ConstantPad2d(padding=padding, value=2.)(v)
            ot.mean().backward()
            ov.mean().backward()

            np.testing.assert_almost_equal(ov.data.numpy(), ot.data.numpy())
            np.testing.assert_almost_equal(v.grad.numpy(), t.grad.data.numpy())
        

if __name__ == '__main__':
    unittest.main(['-m', 'add', '-s'])
