import _init_path
import unittest
import numpy as np 
import time

import pdll as L 


class Testing(unittest.TestCase):

    def test_dataloader(self, ):
        ''' '''

        class DB(L.io.dataset.Dataset):
            def __init__(self, ):
                self.data = np.arange(100)

            def __getitem__(self, idx):
                time.sleep(0.001)
                return self.data[idx]

            def __len__(self, ):
                return len(self.data)

        dataset = DB()
        dataloader = L.io.dataloader.DataLoader(dataset, batch_size=20, num_workers=8, shuffle=True)

        start = time.time()

        for e in range(3):
            for i, batch in enumerate(dataloader):
                time.sleep(0.2)
                print(e, i, len(batch), batch)

        for i, batch in enumerate(dataloader):
            print('---', i, len(batch), batch)

        print('-------------------------------------------------')
        print(f'----------------{time.time()-start}-------------')
        print('-------------------------------------------------')



    # def test_mnist(self, ):
    #     mnist = L.io.dataset.MNIST(train=False)
    #     print('dataloader: ', mnist[0][0].shape)
    #     dataloader = L.io.DataLoader(mnist, batch_size=5000)
    #     for i, data in enumerate(dataloader):
    #         print(i, len(data), len(data[1]), np.array(data[0]).shape)


if __name__ == '__main__':
    unittest.main(['-m', 'add', '-s'])

