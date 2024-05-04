import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from .utils import RGBA2RGB

class MyDataset(Dataset):
    def __init__(self, X, Y, transforms=None):
        self.X = X
        self.Y = Y
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = Image.open(self.X[index])
        X = self.transforms(X)
        Y = self.Y[index]
        return X, Y

def GetTrainingData(root, args, input_shape=(3, 224, 224)):
        if root is None:
            raise ValueError('Data directory not specified!')

        train_trainsforms = transforms.Compose([
            RGBA2RGB(),
            transforms.Resize([224, 224]),
            transforms.RandomCrop(224, padding=24),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        val_trainsforms = transforms.Compose([
            RGBA2RGB(),
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        path_0 = os.path.join(root, 'No_DR/')
        path_1 = os.path.join(root, 'Mild/')
        path_2 = os.path.join(root, 'Moderate/')
        path_3 = os.path.join(root, 'Severe/')
        path_4 = os.path.join(root, 'Proliferate_DR/')

        data_0 = sorted(glob.glob(path_0 + '*'))
        data_1 = sorted(glob.glob(path_1 + '*'))
        data_2 = sorted(glob.glob(path_2 + '*'))
        data_3 = sorted(glob.glob(path_3 + '*'))
        data_4 = sorted(glob.glob(path_4 + '*'))

        data_all = data_0 + data_1 + data_2 + data_3 + data_4
        print('Length of the whole data is: ', len(data_all))

        labels = [0] * len(data_0) + [1] * len(data_1) + [2] * len(data_2) \
                 + [3] * len(data_3) + [4] * len(data_4)

        labels = torch.tensor(labels).long()
        X_train, X_val, y_train, y_val = train_test_split(data_all, labels, test_size=0.1, random_state=5)
        print('Train：', len(X_train), 'Validation：', len(X_val))

        train_dataset = MyDataset(X_train, y_train, train_trainsforms)
        val_dataset = MyDataset(X_val, y_val, val_trainsforms)

        return train_dataset, val_dataset


def GetTestData(root, input_shape=(3, 224, 224)):
    if root is None:
        raise ValueError('Data directory not specified!')

    test_trainsforms = transforms.Compose([
            RGBA2RGB(),
            # CropfromGray(threshold=2),
            transforms.Resize([224, 224]),
            # GaussianFiltered(sigmaX=30),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    path_0 = os.path.join(root, 'No_DR/')
    path_1 = os.path.join(root, 'Mild/')
    path_2 = os.path.join(root, 'Moderate/')
    path_3 = os.path.join(root, 'Severe/')
    path_4 = os.path.join(root, 'Proliferate_DR/')

    data_0 = sorted(glob.glob(path_0 + '*'))
    data_1 = sorted(glob.glob(path_1 + '*'))
    data_2 = sorted(glob.glob(path_2 + '*'))
    data_3 = sorted(glob.glob(path_3 + '*'))
    data_4 = sorted(glob.glob(path_4 + '*'))

    data_all = data_0 + data_1 + data_2 + data_3 + data_4
    print('Length of the whole data is: ', len(data_all))

    labels = [0] * len(data_0) + [1] * len(data_1) + [2] * len(data_2) \
             + [3] * len(data_3) + [4] * len(data_4)
    labels = torch.tensor(labels).long()

    test_dataset = MyDataset(data_all, labels, test_trainsforms)

    return test_dataset
