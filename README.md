# PDLL😊
> including autograd, nn modules, optimizer, .etc.

--- 
## DOCS📖
- [x] mnist

name | performance
---|---
[mnist](./examples/mnist.py) | acc=0.99


## CODE💻

```python
In [11]: import pdll as L
In [12]: a = L.rand(3, 4, requires_grad=True)
In [13]: b = L.rand(4, 7)
In [14]: c = a @ b
In [15]: d = L.rand(7)
In [16]: e = (((c + d) ** 2) - c.mean(axis=-1, keepdims=True)).sum()
In [17]: e.backward()
In [18]: a.grad
Out[18]: 
array([[5.47649723, 8.3762083 , 7.68493488, 9.39777235],
       [5.25936306, 6.63004188, 6.32281295, 6.94581515],
       [4.80345563, 3.93464588, 3.48897623, 3.76960884]])

```
