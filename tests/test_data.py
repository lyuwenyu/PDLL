import _init_path
import unittest
import numpy as np 

import pdll as L 



class Testing(unittest.TestCase):

    def test_data(self, ):
        ''' '''

        class DB(L.io.dataset.Dataset):
            def __init__(self, ):
                self.data = np.arange(100)
            def __getitem__(self, idx):
                return self.data[idx]
            def __len__(self, ):
                return len(self.data)

        dataset = DB()
        dataloader = L.io.dataloader.DataLoader(dataset, batch_size=10, shuffle=True)

        for e in range(3):
            for i, batch in enumerate(dataloader):
                print(e, i, len(batch), batch)


if __name__ == '__main__':
    unittest.main(['-m', 'add', '-s'])

