
from datasets.seq_cifar10 import SequentialCIFAR10
from datasets.seq_cifar100 import SequentialCIFAR100
from datasets.utils.continual_dataset import ContinualDataset
from datasets.seq_mnist import SequentialMNIST
from datasets.seq_tinyimagenet import SequentialTinyImagenet
from argparse import Namespace

NAMES = {
    SequentialCIFAR10.NAME: SequentialCIFAR10,
    SequentialCIFAR100.NAME: SequentialCIFAR100,
    SequentialMNIST.NAME: SequentialMNIST,
    SequentialTinyImagenet.NAME: SequentialTinyImagenet,
}


def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys()
    return NAMES[args.dataset](args)


def get_gcl_dataset(args: Namespace):
    """
    Creates and returns a GCL dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in GCL_NAMES.keys()
    return GCL_NAMES[args.dataset](args)
