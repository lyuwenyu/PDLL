import _init_path
import unittest
import numpy as np 

import pdll as L 



class Testing(unittest.TestCase):

    def test_dataloader(self, ):
        ''' '''

        class DB(L.io.dataset.Dataset):
            def __init__(self, ):
                self.data = np.arange(100)
            def __getitem__(self, idx):
                return self.data[idx]
            def __len__(self, ):
                return len(self.data)

        dataset = DB()
        dataloader = L.io.dataloader.DataLoader(dataset, batch_size=10, shuffle=False)

        for e in range(3):
            for i, batch in enumerate(dataloader):
                print(e, i, len(batch), batch)

        for i, batch in enumerate(dataloader):
            print('----', i, len(batch), batch)


    def test_dataset(self, ):
        
        mnist = L.io.dataset.MNIST(train=False)
        print('dataloader: ', mnist[0][0].shape)
        dataloader = L.io.DataLoader(mnist, batch_size=2000)

        for i, data in enumerate(dataloader):
            print(i, len(data), len(data[1]), np.array(data[0]).shape )


if __name__ == '__main__':
    unittest.main(['-m', 'add', '-s'])

