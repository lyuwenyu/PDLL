import numpy as np 
import math
import random
import time
import os

from .dataset import Dataset

from multiprocessing import Queue, Process, Pool, Lock
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


class DataLoader(object):
    '''dataloader
    '''
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool=False, num_workers=0, drop_last: bool=True):
        assert batch_size > 0, ''
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.indices = np.arange(len(dataset))
        self._iterator = None

    def __iter__(self):
        '''
        '''
        if self._iterator is None:
            self._iterator = _BaseDataLoaderIter(self)
        else:
            self._iterator.reset()

        return self._iterator

    def __len__(self, ):
        return len(self.dataset) // self.batch_size


class _BaseDataLoaderIter(object):
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.dataset = loader.dataset
        self.indices = loader.indices
        self.batch_size = loader.batch_size
        self.drop_last = loader.drop_last
        self.batch_idx = [-1]
        self.shuffle = loader.shuffle
        self.num_workers = loader.num_workers

        if self.shuffle:
            random.shuffle(self.indices)
        
        if self.num_workers > 0:
            self.executor = ThreadPoolExecutor(max_workers=8)
            
    def __iter__(self, ):
        return self

    def __next__(self, ):
        '''
        '''
        self.batch_idx[0] = self.batch_idx[0] + 1 
        if self.batch_idx[0] >= len(self.indices) // self.batch_size:
            raise StopIteration

        idx = self.indices[self.batch_idx[0] * self.batch_size: (self.batch_idx[0] + 1) * self.batch_size]    
            
        if self.num_workers > 0:
            batch = list(self.executor.map(self.loader.dataset.__getitem__, idx))
        else:
            batch = [self.loader.dataset[i] for i in idx]

        if not isinstance(batch[0], tuple):
            return batch
        else:
            batch = list(zip(*batch))
            return batch

        return batch


    def reset(self, ):
        self.batch_idx[0] = -1
        if self.shuffle:
            random.shuffle(self.indices)
        
"""
    def __init_processes(self, ):

        self.lock = Lock()
        self.queue = Queue(64)

        self._processes = []
        for _ in range(self.num_workers):
            self._processes.append(PreFetcher(self))
            self._processes[-1].start()
            time.sleep(0.1)

        def clearup():
            for p in self._processes:
                p.terminate()
                p.join()
        import atexit
        atexit.register(clearup)

        if self.num_workers > 0:
            for p in self._processes:
                if p.is_alive():
                    p.terminate(); p.join()

        if self.num_workers > 0:
            batch = self.queue.get()
            if all([not p.is_alive() for p in self._processes] + [self.queue.empty()]):
                raise StopIteration

"""

class PreFetcher(Process):
    '''docstring for Fetcher
    '''
    def __init__(self, loaderiter):
        super().__init__()
        self.loaderiter = loaderiter
        self.batch_idx = -1
        self.batch_size = loaderiter.batch_size
        self.indices = np.arange(len(loaderiter.dataset))
        random.shuffle(self.indices)

    def run(self, ):
        while True:
            batch = self.get_next()
            if batch is None:
                break
            self.loaderiter.queue.put(batch)

        self.loaderiter.finished = True

    def get_next(self, ):
        '''
        '''
        self.batch_idx += 1

        if self.batch_idx >= len(self.indices) // self.loaderiter.batch_size:
            return None

        idx = self.indices[self.batch_idx * self.batch_size: (self.batch_idx + 1) * self.batch_size]
        batch = [self.loaderiter.dataset[i] for i in idx]

        return batch