from Dataset import CarlaDataset
from ResNet import Encoder, Binary, Decoder, Autoencoder, Bottleneck

import numpy as np
import csv
import pandas as pd

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import yaml
import re


def get_loss_values():

    loss_and_epoch_values = {
    "best_valid_loss_one" : float('inf'),
    "best_valid_loss_two" : float('inf'),

    "train_loss_avg" : [],
    "val_loss_avg" : [],
    "val_loss_avg_two" : [],

    "best_epoch_same" : 0,
    "best_epoch_other" : 0
    }

    return loss_and_epoch_values
#### Telegram ####
import requests

def tg_sendMessage(msg):
    assert(isinstance(msg, str))
    bot_token = '5012425653:AAHx4O6mzJCv90wundw7F7BKBBWSWgpRxgs'
    bot_chatID = '313580257'
    send_url = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + msg

    response = requests.get(send_url)

    return response.json

def notifyTrainFinish(name):
    assert(isinstance(name, str))
    tg_sendMessage('Training finished for '+name)

#### Data Loader #####

loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))

def get_dataloaders(train_set, val_set_one, val_set_two, batch_s):

  dataset_train = CarlaDataset(train_set, 'pred_depth/', 'opt_flow/')
  dataset_val = CarlaDataset(val_set_one, 'pred_depth/', 'opt_flow/')
  dataset_val_two = CarlaDataset(val_set_two, 'pred_depth/', 'opt_flow/')

  import torchvision.transforms as transforms
  from torch.utils.data import DataLoader
  from torch.utils.data.sampler import SubsetRandomSampler

  train_dataset_size = len(dataset_train)
  train_indices = list(range(train_dataset_size))

  val_dataset_size = len(dataset_val)
  val_indices = list(range(val_dataset_size))

  val_dataset_size_two = len(dataset_val_two)
  val_indices_two = list(range(val_dataset_size_two))

  train_sampler = SubsetRandomSampler(train_indices)
  valid_sampler = SubsetRandomSampler(val_indices)
  valid_sampler_two = SubsetRandomSampler(val_indices_two)

  train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_s, sampler=train_sampler)
  valid_dataloader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_s, sampler=valid_sampler)
  valid_dataloader_two = torch.utils.data.DataLoader(dataset_val_two, batch_size=batch_s, sampler=valid_sampler_two)

  return train_dataloader, valid_dataloader, valid_dataloader_two

def create_model(configs, device):

  encoder = Encoder(Bottleneck, [3, 4, 6, 3]).to(device)
  encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False).to(device)
  encoder.fc = nn.Linear(65536, configs['zsize']).to(device)
  encoder=encoder.to(device)

  #binary = Binary()

  decoder = Decoder().to(device)

  autoencoder = Autoencoder(encoder).to(device)

  from torchsummary import summary

  print(summary(autoencoder,(1,192,640)))

  return autoencoder

def create_output_images(output_dir, model, dataloader, device):
  model.eval()
  images_original, optical_flow = iter(dataloader).next()
  with torch.no_grad():
    print(images_original.shape)
    images = model(images_original.to(device))
    images = images.cpu()
    print(images.shape)
    for i in range(0,images.shape[0]):
      output_tensor = torch.zeros(3,192*3,640)
      image_tensor = images[i,:,:,:]

      label_tensor = optical_flow[i,:,:,:]
      depth_tensor = images_original[i,:,:,:]

      output_tensor[:,:192,:] = images[i,:,:,:]
      output_tensor[:,192:2*192,:] = optical_flow[i,:,:,:]
      output_tensor[:,2*192:,:] = images_original[i,:,:,:]
      output = transforms.ToPILImage()(output_tensor).convert("RGB")
      output.save(output_dir + "/frame_" + str(i) + ".png")

def store_model(model_version, model, loss_values, configs,epoch):
  if model_version == 'Same':
    if loss_values['val_loss_avg'][-1] < loss_values['best_valid_loss_one']:
      loss_values['best_valid_loss_one'] = loss_values['val_loss_avg'][-1]
      torch.save(model.state_dict(),'ResNet18_Same_Dataset.pt')
      loss_values['best_epoch_same'] = epoch
  elif model_version == 'Other':
    if loss_values['val_loss_avg_two'][-1] < loss_values['best_valid_loss_two']:
      loss_values['best_valid_loss_two'] = loss_values['val_loss_avg_two'][-1]
      torch.save(model.state_dict(),'ResNet18_Other_Dataset.pt')
      loss_values['best_epoch_other'] = epoch
  elif model_version == 'Final':
    torch.save(model.state_dict(),'ResNet18_' + str(configs['num_epochs']) + 'E_Dataset.pt')
  else:
    print('model ' + model_version + ' unknown!')