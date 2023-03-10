
import torch.nn as nn
import torch
from argparse import Namespace


class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, args: Namespace) -> None:
        super(ContinualModel, self).__init__()

        self.args = args
        self.device = self.args.device

    def begin_il(self, dataset) -> torch.Tensor:
        pass

    def end_il(self, dataset) -> float:
        pass
