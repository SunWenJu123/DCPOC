
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from abc import abstractmethod
from argparse import Namespace
from torch.utils.data import DataLoader, TensorDataset, Dataset
from typing import Tuple
from torchvision import datasets
import numpy as np
from pathlib import Path

class ContinualDataset:
    NAME = None
    SETTING = None

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.test_loaders = []
        self.i = 0
        self.args = args
        self.nt = args.nt
        self.nc = args.nc

        self.t_c_arr = args.t_c_arr if args.t_c_arr else self.get_balance_classes()

    def get_balance_classes(self):
        class_arr = list(range(self.nc))
        cpt = self.nc // self.nt
        return [class_arr[i:i+cpt] for i in range(0, len(class_arr), cpt)]

    @abstractmethod
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        pass

    @abstractmethod
    def not_aug_dataloader(self, batch_size: int) -> DataLoader:
        """
        Returns the dataloader of the current task,
        not applying data augmentation.
        :param batch_size: the batch size of the loader
        :return: the current training loader
        """
        pass



def store_masked_loaders(train_dataset: datasets, test_dataset: datasets,
                    setting: ContinualDataset) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    c_arr = setting.t_c_arr[setting.i]
    train_mask = np.logical_and(np.array(train_dataset.targets) >= c_arr[0],
                                np.array(train_dataset.targets) <= c_arr[-1])
    test_mask = np.logical_and(np.array(test_dataset.targets) >= c_arr[0],
                               np.array(test_dataset.targets) <= c_arr[-1])

    train_dataset.data = train_dataset.data[train_mask]
    test_dataset.data = test_dataset.data[test_mask]

    train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    train_loader = DataLoader(train_dataset,
                              batch_size=setting.args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset,
                             batch_size=setting.args.batch_size, shuffle=False, num_workers=4)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    setting.i += 1
    return train_loader, test_loader


def get_previous_train_loader(train_dataset: datasets, batch_size: int,
                              setting: ContinualDataset) -> DataLoader:
    """
    Creates a dataloader for the previous task.
    :param train_dataset: the entire training set
    :param batch_size: the desired batch size
    :param setting: the continual dataset at hand
    :return: a dataloader
    """
    train_mask = np.logical_and(np.array(train_dataset.targets) >=
        setting.i - setting.N_CLASSES_PER_TASK, np.array(train_dataset.targets)
        < setting.i - setting.N_CLASSES_PER_TASK + setting.N_CLASSES_PER_TASK)

    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


class ILDataset(Dataset):  # 继承Dataset
    def __init__(self, feature, label, feature_transform=None, label_transform=None):  # __init__是初始化该类的一些基础参数
        self.feature = feature
        self.label = label
        self.attributes = ['feature', 'label']
        self.transforms = [feature_transform, label_transform]

    def __len__(self):  # 返回整个数据集的大小
        return self.feature.shape[0]

    def set_att(self, att_name, att_data, att_transform=None):  # 设置一个需要代码中需要的属性
        self.attributes.append(att_name)
        self.transforms.append(att_transform)
        setattr(self, att_name, att_data)

    def get_att_names(self):
        return self.attributes

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        ret_tuple = ()
        for i, att in enumerate(self.attributes):
            att_data = getattr(self, att)[index]

            transform = self.transforms[i]
            if transform:
                att_data = transform(att_data)

            ret_tuple += (att_data,)
        return ret_tuple

def getfeature_loader(train_dataset: datasets, test_dataset: datasets, setting: ContinualDataset):

    my_file = Path("data/" + setting.args.dataset + "-" + setting.args.featureNet + "-train-data.npy")
    if my_file.exists():
        print("feature already extracted")
        train_data = np.load("data/" + setting.args.dataset + "-" + setting.args.featureNet + "-train-data.npy", allow_pickle=True)
        train_label = np.load("data/" + setting.args.dataset + "-" + setting.args.featureNet + "-train-label.npy", allow_pickle=True)
        test_data = np.load("data/" + setting.args.dataset + "-" + setting.args.featureNet + "-test-data.npy", allow_pickle=True)
        test_label = np.load("data/" + setting.args.dataset + "-" + setting.args.featureNet + "-test-label.npy", allow_pickle=True)
    else:
        print("feature file not found !!  extracting feature ...")
        test_data, test_label = get_feature_by_extractor(test_dataset, setting.extractor, setting)
        train_data, train_label = get_feature_by_extractor(train_dataset, setting.extractor, setting)

        np.save("data/" + setting.args.dataset + "-" + setting.args.featureNet + "-train-data.npy", train_data, allow_pickle=True)
        np.save("data/" + setting.args.dataset + "-" + setting.args.featureNet + "-train-label.npy", train_label, allow_pickle=True)
        np.save("data/" + setting.args.dataset + "-" + setting.args.featureNet + "-test-data.npy", test_data, allow_pickle=True)
        np.save("data/" + setting.args.dataset + "-" + setting.args.featureNet + "-test-label.npy", test_label, allow_pickle=True)


    c_arr = setting.t_c_arr[setting.i]
    train_mask = np.logical_and(np.array(train_label) >= c_arr[0],
                                np.array(train_label) <= c_arr[-1])
    test_mask = np.logical_and(np.array(test_label) >= c_arr[0],
                               np.array(test_label) <= c_arr[-1])

    train_data = train_data[train_mask]
    test_data = test_data[test_mask]

    train_label = torch.from_numpy(train_label[train_mask])
    test_label = torch.from_numpy(test_label[test_mask])

    train_dataset = ILDataset(train_data, train_label, feature_transform=setting.train_transform)
    test_dataset = ILDataset(test_data, test_label, feature_transform=setting.test_transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=setting.args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset,
                             batch_size=setting.args.batch_size, shuffle=False, num_workers=4)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    setting.i += 1
    return train_loader, test_loader

def get_feature_by_extractor(train_dataset: datasets, extractor, setting: ContinualDataset):
    if extractor:
        extractor = extractor.to(setting.args.device).eval()
    train_loader = DataLoader(train_dataset,
                              batch_size=256, shuffle=False, num_workers=4)
    features, labels = [], []
    for data in train_loader:
        # print(data)
        img = data[0]
        label = data[1]
        img = img.to(setting.args.device)

        if extractor:
            with torch.no_grad():
                feature = extractor(img)
        else:
            feature = img

        feature = feature.to('cpu')
        img = img.to('cpu')

        features.append(feature)
        labels.append(label)

    feature = torch.cat(features).numpy()
    label = torch.cat(labels).numpy()

    return feature, label