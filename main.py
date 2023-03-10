
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from argparse import ArgumentParser
from utils.training import train_il
from utils.conf import set_random_seed
import torch


def main():
    parser = ArgumentParser(description='DCPOC', allow_abbrev=False)
    args = parser.parse_known_args()[0]
    args.model = 'dcpoc'
    args.seed = None
    args.validation = False
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.img_dir = 'img/dcpoc/'
    if not os.path.exists(args.img_dir):
        os.makedirs(args.img_dir)

    # seq-cifar10 with PTF
    args.dataset = 'seq-tinyimg'
    args.featureNet = 'vgg11'
    args.nt = 5
    args.t_c_arr = None

    args.lr = 2e-4
    args.batch_size = 128
    args.n_epochs = 1

    args.lambda2 = 0.2
    args.kld_ratio = 0.25
    args.eps = 1
    args.embedding_dim = 128  # latent space dim
    args.weight_decay = 1e-2
    args.lambda1 = 1
    args.r_inter = 500
    args.r_intra = 10

    # seq-cifar100  with PTF
    # args.dataset = 'seq-cifar100'
    # args.featureNet = 'vgg11'
    # args.lr = 1e-4
    # args.batch_size = 64
    # args.n_epochs = 50
    #
    # args.lambda2 = 0.5
    # args.kld_ratio = 0
    # args.eps = 1
    # args.embedding_dim = 250
    # args.weight_decay = 0.001
    # args.lambda1 = 0.1
    # args.r_inter = 100
    # args.r_intra = 0
    # args.isPseudo = True
    # args.isPrint = False
    # args.nf = 64

    # seq-tinyimg  with PTF
    # args.dataset = 'seq-tinyimg'
    # args.featureNet = 'vgg11'
    # args.lr = 5e-5
    # args.batch_size = 32
    # args.n_epochs = 100
    #
    # args.lambda2 = 10
    # args.kld_ratio = 0
    # args.eps = 1
    # args.embedding_dim = 250
    # args.weight_decay = 0.001
    # args.lambda1 = 0.01
    # args.r_inter = 700
    # args.r_intra = 0
    # args.isPseudo = True
    # args.isPrint = False
    # args.nf = 64

    if args.seed is not None:
        set_random_seed(args.seed)

    train_il(args)

if __name__ == '__main__':
    main()
