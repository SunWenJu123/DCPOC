# DCPOC: Exemplar-free Class Incremental Learning via Discriminative and Comparable Parallel One-class Classifiers

## Paper

Official repository of Exemplar-free Class Incremental Learning via Discriminative and Comparable Parallel One-class Classifiers

>   Journal: Pattern Recognition

## Setup

-   Use `./utils/main.py` to run experiments.
-   Some training result can be found in folder `./result`.

## Datasets

**Class-IL / Task-IL settings**

-   Sequential MNIST
-   Sequential CIFAR-10
-   Sequential CIFAR-100
-   Sequential Tiny ImageNet

## Requirement

+ numpy==1.16.4
+ Pillow==6.1.0
+ torch==1.3.1
+ torchvision==0.4.2

## Citation

```
@article{
    DCPOC,
    title = {Exemplar-free class incremental learning via discriminative and comparable parallel one-class classifiers},
    journal = {Pattern Recognition},
    volume = {140},
    pages = {109561},
    year = {2023},
    issn = {0031-3203},
    doi = {https://doi.org/10.1016/j.patcog.2023.109561},
    url = {https://www.sciencedirect.com/science/article/pii/S0031320323002613},
    author = {Wenju Sun and Qingyong Li and Jing Zhang and Danyu Wang and Wen Wang and YangLi-ao Geng},
}
```

## Related repository

https://github.com/aimagelab/mammoth