import _init_path

import pdll as L

a = L.randn(2, 3, requires_grad=True)
b = L.randn(3, 4)
c = (a @ b + L.randn(4)) ** 2

c.mean().backward()

print(c.data)


a.requires_grad = False

