# PDLL😊
> including autograd, nn modules, optimizer, io, .etc.


## Getting Started

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
--- 
- [More Examples](./examples/)

Name | Performance | Commits 
---|---|---
[mnist](./examples/mnist.py) | acc=0.99 | an exmaple midified from pytorch mnist, but with new network achitecture.
---

## About PDLL

PDLL is a python deep learning library.

Module | Description
---|---
[pdll.backend]() | a tensor library, like numpy or others.
[pdll.autograd]() | an automatic differentiation library, that record operations on variable type. 
[pdll.nn]() | a neural network library based on autograd
[pdll.optim]() | an optimizer library for deep learning
[pdll.io]() | dataset, dataloader and serialization

---
To learn more about contributing to PDLL, please contact me.

## About Me
 - Email