import torch
from torch import nn
import matplotlib.pyplot as plt
from d2l import torch as d2l
import torchvision

d2l.DATA_HUB['pokemon'] = (d2l.DATA_URL + 'pokemon.zip',
                          'c065c0e2593b8b161a2d7873e42418bf6a21106c')

data_dir = d2l.download_extract("pokemon",)
pokemon = torchvision.datasets.ImageFolder(data_dir)
d2l.show_images