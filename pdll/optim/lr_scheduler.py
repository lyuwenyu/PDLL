import math

class _Scheduler(object):
    
    def __init__(self, optimizer, epochs=-1):
        self._optimizer = optimizer
        self._epochs = -1

    def step(self, ):
        raise NotImplementedError


class MultiStepLR(_Scheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, epochs=-1):
        super().__init__(optimizer, epochs)
        self._gamma = gamma
        self._milestones = milestones
        self._lr = optimizer.lr 

    def step(self, ):
        '''
        '''
        self._epochs += 1
        if self._epochs in self._milestones:
            self._lr *= self._gamma
        
        self._optimizer.lr = self._lr


    def state_dict(self, ):
        '''
        '''
        raise NotImplementedError