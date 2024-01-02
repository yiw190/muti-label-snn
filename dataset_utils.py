import numpy as np
from sklearn.model_selection import train_test_split   
import torch
from torch.utils.data import Dataset, DataLoader
def wgn(x, snr):
    b, h, w = x.shape
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / (b*h*w)
    npower = xpower / snr
    return np.random.randn(b, h, w) * np.sqrt(npower)


def add_noise(data, snr_num):
    data_p = data
    rand_data = wgn(data, snr_num)
    data_n = data_p + rand_data
    return data_n

def load_data(file):
    data = np.load(file)
    trainx = data['arr_0']  # (38015, 52. 52)
    trainx = add_noise(trainx, 4)
    trainy = data['arr_1']  # (38015, 8), 0-1999: C24, 34866-35014: C9, 33000-33865: C7
    x_train, x_test, y_train, y_test = train_test_split(trainx, trainy, test_size=0.2, random_state=10)
    return x_train, x_test, y_train, y_test 

class TrainSet(Dataset):
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def __getitem__(self, index):
        train_data = torch.tensor(self.x_train[index])
        train_label = torch.tensor(self.y_train[index])
        return train_data, train_label

    def __len__(self):
        return self.x_train.shape[0]

class TestSet(Dataset):
    def __init__(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test

    def __getitem__(self, index):
        test_data = torch.tensor(self.x_test[index])
        test_label = torch.tensor(self.y_test[index])
        return test_data, test_label

    def __len__(self):
        return self.x_test.shape[0]