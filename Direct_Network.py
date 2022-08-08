import torch
from torch.autograd import Variable
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models,transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np 
import os
from torch.autograd import Function
from collections import OrderedDict
import torch.nn as nn
import math

from ResNet import Encoder, Binary, Decoder, Autoencoder, Bottleneck
from Action_Predicter_Models import Action_Predicter_Dense

from utils import *
from my_logging import *
from train_and_val import *

class Direct_Network (nn.Module):
    def __init__(self, configs, device):
        super(Direct_Network, self).__init__()
        self.autoencoder = create_model(configs, device)
        self.predicter = Action_Predicter_Dense()
        self.name = 'DIRECT'

    def forward(self, x):
    	#print(x.shape)

    	embedding = self.autoencoder(x)
    	#print('-----------')
    	#print(embedding.shape)
    	#embedding.transpose_(1, 2)
    	#print(embedding.shape)

    	steering = self.predicter(embedding)

    	return steering