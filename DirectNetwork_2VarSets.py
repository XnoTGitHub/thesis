import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image

import numpy as np
import csv
import pandas as pd
import argparse
import yaml
import re

from Dataset import CarlaDataset
from ResNet import Encoder, Binary, Decoder, Autoencoder, Bottleneck
from Action_Predicter_Models import Action_Predicter_Dense
from Direct_Network import Direct_Network

from utils import *
from my_logging import *
from train_and_val import *

loss_values = get_loss_values()

parser = argparse.ArgumentParser(description="Direct")
parser.add_argument("--config", help="YAML config file")
args = parser.parse_args()

with open(args.config) as f:
  configs = yaml.load(f,Loader=loader)

###################
#### DataLoaer ####

train_dataloader, valid_dataloader, valid_dataloader_two = get_dataloaders(configs['Name'], configs['TRAIN_SET'], configs['VAL_SET'], configs['VAL_SET_TWO'], configs['batch_size'])

################
#### Model #####

device = torch.device("cuda:0" if (torch.cuda.is_available() and configs['use_gpu']) else "cpu")

#if configs['Name'] == 'DEPTH' or configs['Name'] == 'VAR' or configs['Name'] == 'DIRECT':
#  autoencoder = create_model(configs, device)########sollte hier num_input_channels gleich 1 sein??!
#elif configs['Name'] == 'RGB':
#  autoencoder = create_model(configs, device, num_input_channels=3)
#print(configs['Name'])
autoencoder = Direct_Network(configs, device).to(device)
num_params = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
print('Number of parameters: %d' % num_params)

TRAIN = configs['TRAIN']
if (TRAIN):
  print('Training has started..')
  log_header(configs,autoencoder)

  optimizer = torch.optim.Adam(params=autoencoder.parameters() , lr=configs['learning_rate'], weight_decay=configs['weight_decay'])

  # set to training mode
  autoencoder.train()

  print('Training ...')
  for epoch in range(configs['num_epochs']):
      loss_values['train_loss_avg'].append(0)
      loss_values['val_loss_avg'].append(0)
      loss_values['val_loss_avg_two'].append(0)

      ### Training ###

      loss_values['train_loss_avg'][-1] = train(train_dataloader, autoencoder, optimizer, device)

      ### First Validation ###

      loss_values['val_loss_avg'][-1] = validate(valid_dataloader, autoencoder, optimizer, device)
      store_model('Same_' + configs['Name'],autoencoder, loss_values, configs, epoch)

      ### Second Validation ###

      loss_values['val_loss_avg_two'][-1] = validate(valid_dataloader_two, autoencoder, optimizer, device)
      store_model('Other_' + configs['Name'],autoencoder, loss_values, configs, epoch)

      log_epoch(configs, loss_values['train_loss_avg'][-1], loss_values['val_loss_avg'][-1], loss_values['val_loss_avg_two'][-1], epoch)
      store_model('Final_'+ configs['Name'], autoencoder, loss_values, configs, epoch)
  notifyTrainFinish('Direct_Network')
else:
  #if configs['Name'] == 'VAR':
  autoencoder.load_state_dict(torch.load('ResNet18_Final_' + configs['Name'] + '_' + str(configs['num_epochs']) + 'E_Dataset.pt'))
  #else:
  #  autoencoder.load_state_dict(torch.load('ResNet18_Final_' + configs['Name'] + 'E_Dataset.pt'))

if(TRAIN):
  log_best(configs, loss_values['best_epoch_same'], loss_values['best_epoch_other'])

### Visualize output ###

create_dirs('out/autoencoder_output_final','out/autoencoder_output_same','out/autoencoder_output_other')

create_output_images('thesis/out/autoencoder_output_final', autoencoder, valid_dataloader, device, configs)
# Dataset Same
autoencoder.load_state_dict(torch.load('ResNet18_Same_' + configs['Name'] + '_Dataset.pt'))
create_output_images('thesis/out/autoencoder_output_same', autoencoder, valid_dataloader, device, configs)
# Dataset Same
autoencoder.load_state_dict(torch.load('ResNet18_Other_'+ configs['Name'] + '_Dataset.pt'))
create_output_images('thesis/out/autoencoder_output_other', autoencoder, valid_dataloader, device, configs)

print('finish')