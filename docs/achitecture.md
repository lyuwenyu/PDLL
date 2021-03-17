
The Achitecture of library.

```
├── pdll
│   ├── __init__.py
│   ├── autograd
│   │   ├── __init__.py
│   │   ├── backpropag.py
│   │   ├── creation.py
│   │   ├── function.py
│   │   ├── operator.py
│   │   ├── tensor.py
│   │   └── utils.py
│   ├── backend
│   │   ├── __init__.py
│   │   ├── executor.py
│   │   ├── operator.py
│   │   └── storage.proto
│   ├── io
│   │   ├── __init__.py
│   │   ├── dataloader.py
│   │   ├── dataset.py
│   │   └── serializable.py
│   ├── nn
│   │   ├── __init__.py
│   │   ├── functional.py
│   │   ├── initialization.py
│   │   ├── modules
│   │   │   ├── __init__.py
│   │   │   ├── activation.py
│   │   │   ├── attention.py
│   │   │   ├── convolution.py
│   │   │   ├── dropout.py
│   │   │   ├── linear.py
│   │   │   ├── loss.py
│   │   │   ├── module.py
│   │   │   ├── normalization.py
│   │   │   ├── padding.py
│   │   │   └── pooling.py
│   │   ├── parameter.py
│   │   └── utils.py
│   └── optim
│       ├── __init__.py
│       ├── lr_scheduler.py
│       └── optimizer.py
```