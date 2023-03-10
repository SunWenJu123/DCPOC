
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils.conf import base_path
from PIL import Image
import numpy as np
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, getfeature_loader
from typing import Tuple
from argparse import Namespace


class MyMNIST(MNIST):
    """
    Overrides the MNIST dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.ToTensor()
        super(MyMNIST, self).__init__(root, train,
                                      transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[type(Image), int]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class SequentialMNIST(ContinualDataset):

    NAME = 'seq-mnist'
    SETTING = 'class-il'

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        args.nc = 10
        super(SequentialMNIST, self).__init__(args)

        self.extractor = None
        self.normalization_transform = None
        self.dnormalization_transform = None
        self.train_transform = None
        self.test_transform = None

    def get_data_loaders(self):
        transform = transforms.ToTensor()
        train_dataset = MyMNIST(base_path() + 'MNIST',
                                train=True, download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        transform, self.NAME)
        else:
            test_dataset = MNIST(base_path() + 'MNIST',
                                train=False, download=True, transform=transform)

        train, test = getfeature_loader(train_dataset, test_dataset, self)
        return train, test

    def not_aug_dataloader(self, batch_size):
        transform = transforms.ToTensor()
        train_dataset = MyMNIST(base_path() + 'MNIST',
                                train=True, download=True, transform=transform)
        train_mask = np.logical_and(np.array(train_dataset.targets) >= self.i -
            self.N_CLASSES_PER_TASK, np.array(train_dataset.targets) < self.i)

        train_dataset.data = train_dataset.data[train_mask]
        train_dataset.targets = np.array(train_dataset.targets)[train_mask]

        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size, shuffle=True)
        return train_loader
