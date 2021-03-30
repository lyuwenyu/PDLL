import _init_path

import pdll as L 
import torch
import numpy as np


import pdll as L

a = L.rand(2, 2, 3, requires_grad=True)
b = L.rand(3, 3)
c = a @ b + L.rand(2, 3)
d = (c ** 2) * 2 - L.ones_like(c)
d.mean().backward()

print(a.grad.shape)