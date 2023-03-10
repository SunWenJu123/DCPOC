
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
import torchvision.models as models
from utils.conf import base_path
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, getfeature_loader
from datasets.utils.continual_dataset import get_previous_train_loader
from typing import Tuple
from argparse import Namespace
from datasets.transforms.denormalization import DeNormalize


class MyCIFAR100(CIFAR100):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """

    def __init__(self, root, train=True, transform=None, target_transform=None) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.transform = transform
        self.target_transform = target_transform
        super(MyCIFAR100, self).__init__(root, train, transform, target_transform, download=True)

    def __getitem__(self, index: int) -> Tuple[type(Image), int]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class SequentialCIFAR100(ContinualDataset):
    NAME = 'seq-cifar100'
    SETTING = 'class-il'

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        args.nc = 100
        super(SequentialCIFAR100, self).__init__(args)

        if self.args.featureNet:
            if self.args.featureNet == 'resnet18':
                self.extractor = models.resnet18(pretrained=True)
            elif self.args.featureNet == 'vgg11':
                self.extractor = models.vgg11(pretrained=True)
            elif self.args.featureNet == 'resnet34':
                self.extractor = models.resnet34(pretrained=True)

            self.normalization_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.dnormalization_transform = DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.cifar10_train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                self.normalization_transform])
            self.cifar10_test_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                self.normalization_transform])
            self.train_transform = None
            self.test_transform = None
        else:
            self.normalization_transform = None
            self.dnormalization_transform = None
            self.cifar10_train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])
            self.cifar10_test_transform = transforms.Compose([transforms.ToTensor()])
            self.train_transform = transforms.Compose([transforms.ToPILImage(), self.cifar10_train_transform])
            self.test_transform = transforms.Compose([transforms.ToPILImage(), self.cifar10_test_transform])

    def get_data_loaders(self):

        train_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=True, transform=self.cifar10_train_transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        self.cifar10_test_transform, self.NAME)
        else:
            test_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=False, transform=self.cifar10_test_transform)

        train, test = getfeature_loader(train_dataset, test_dataset, setting=self)
        return train, test

    def not_aug_dataloader(self, batch_size):
        transform = transforms.Compose(
            [transforms.Resize(224), transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=True, transform=transform)
        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader
