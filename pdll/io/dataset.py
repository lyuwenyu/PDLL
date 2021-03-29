import struct
import numpy as np 
import gzip
import os

class Dataset(object):
    '''dataset
    '''
    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self, ):
        raise NotImplementedError

    
    
class TensorDataset(Dataset):

    def __init__(self, data):
        self.data = data 
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self, ):
        return self.data.shape[0]


class MNIST(Dataset):

    def __init__(self, root='../data/MNIST/raw/', train=True):
        '''
        '''
        train_images_idx3_ubyte_file = os.path.join(root, 'train-images-idx3-ubyte.gz')
        train_labels_idx1_ubyte_file = os.path.join(root, 'train-labels-idx1-ubyte.gz')
        test_images_idx3_ubyte_file = os.path.join(root, 't10k-images-idx3-ubyte.gz')
        test_labels_idx1_ubyte_file = os.path.join(root, 't10k-labels-idx1-ubyte.gz')
        
        if train:
            images, labels = self._load_mnist(train_images_idx3_ubyte_file, train_labels_idx1_ubyte_file)
        else:
            images, labels = self._load_mnist(test_images_idx3_ubyte_file, test_labels_idx1_ubyte_file)
        
        self.images = self._normalize_image(images).reshape(-1, 1, 28, 28)
        self.labels = labels
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self, ):
        return len(self.labels)

    def _load_mnist(self, data_file, label_file):
        data = self._read_image(data_file)
        label = self._read_label(label_file)
        return data, label

    def _read_image(self, path):
        with gzip.open(path, 'rb') as f:
            _, num, rows, cols = struct.unpack('>4I', f.read(16))
            img=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
        return img

    def _read_label(self, path):
        with gzip.open(path, 'rb') as f:
            _, num = struct.unpack('>2I', f.read(8))
            lab = np.frombuffer(f.read(), dtype=np.uint8)
        return lab
    
    def _normalize_image(self, image):
        img = image.astype(np.float32) / 255.0
        return img



if __name__ == '__main__':

    mnist = MNIST()
    print(mnist[0][0].shape)
    print(len(mnist))