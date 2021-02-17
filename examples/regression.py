import pdll as L 
import numpy as np 
import matplotlib.pyplot as plt

_data = np.pi * (np.random.rand(10000, 1) * 2 - 1)
_label = np.cos(_data) + 2 

class Model(L.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.l1 = L.nn.Linear(1, 32)
        self.l2 = L.nn.Linear(32, 32)
        self.l3 = L.nn.Linear(32, 1)
        
    def forward(self, data):
        out = self.l1(data).tanh()
        out = self.l2(out).tanh()
        out = self.l3(out)        
        return out

def train(model):
    ''''''
    data = L.from_numpy(_data)
    label = L.from_numpy(_label)
    lr = 0.01

    for i in range(500):
        
        model.zero_grad()
        out = model(data)
        loss = ((out - label) ** 2).mean()
        loss.backward()
        
        for p in model.parameters():
            p.data -= p.grad * lr
            
        if i % 200 == 0:
            print(i, loss.data)

model = Model()
train(model)

plt.scatter(_data, model(L.Variable(_data)).data)
plt.show()