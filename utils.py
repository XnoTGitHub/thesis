from Dataset import CarlaDataset
from ResNet import Encoder, Binary, Decoder, Autoencoder, Bottleneck, Var_Autoencoder, Var_Encoder, DirectEncoder

import numpy as np
import csv
import pandas as pd

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os

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

def get_dataloaders(model_version,train_set, val_set_one, val_set_two, batch_s):

  if 'DEPTH' in model_version or 'VAR' in model_version:

    dataset_train = CarlaDataset(train_set, 'pred_depth/', 'ofl/')
    dataset_val = CarlaDataset(val_set_one, 'pred_depth/', 'ofl/')
    dataset_val_two = CarlaDataset(val_set_two, 'pred_depth/', 'ofl/')

  elif model_version == 'RGB':

    dataset_train = CarlaDataset(train_set, 'rgb/', 'ofl/')
    dataset_val = CarlaDataset(val_set_one, 'rgb/', 'ofl/')
    dataset_val_two = CarlaDataset(val_set_two, 'rgb/', 'ofl/')

  elif 'DIRECT_rgb' in model_version:

    dataset_train = CarlaDataset(train_set, 'rgb/', 'direct')
    dataset_val = CarlaDataset(val_set_one, 'rgb/', 'direct')
    dataset_val_two = CarlaDataset(val_set_two, 'rgb/', 'direct')

  elif 'DIRECT' in model_version:

    dataset_train = CarlaDataset(train_set, 'pred_depth/', 'direct')
    dataset_val = CarlaDataset(val_set_one, 'pred_depth/', 'direct')
    dataset_val_two = CarlaDataset(val_set_two, 'pred_depth/', 'direct')


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

def create_model(configs, device,num_input_channels=1):


  if 'DEPTH' in configs['Name'] or 'RGB' in configs['Name']:

    encoder = Encoder(Bottleneck, [3, 4, 6, 3],num_input_channels).to(device)
    encoder.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False).to(device)
    encoder.fc = nn.Linear(65536, configs['zsize']).to(device)
    encoder=encoder.to(device)

    #binary = Binary()

    decoder = Decoder().to(device)

    autoencoder = Autoencoder(encoder).to(device)

    #from torchsummary import summary

    #print(summary(autoencoder,(num_input_channels,192,640)))

    return autoencoder

  elif configs['Name'] == 'VAR':

    encoder = Var_Encoder(Bottleneck, [3, 4, 6, 3],num_input_channels).to(device)
    encoder.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False).to(device)
    #encoder.fc = nn.Linear(65536, configs['zsize']).to(device)
    encoder.fc_mu = nn.Linear(65536, configs['zsize']).to(device)
    encoder.fc_logvar = nn.Linear(65536, configs['zsize']).to(device)
    encoder.fc_logvar.weight.data.normal_(-0.001, 0.001)
    encoder=encoder.to(device)

    #binary = Binary()

    decoder = Decoder().to(device)

    autoencoder = Var_Autoencoder(encoder).to(device)

    from torchsummary import summary

    print(summary(autoencoder,(num_input_channels,192,640)))

    return autoencoder
  elif 'DIRECT' in configs['Name']:
    print('DIRECT_________')
    num_input_channels = 3
    if 'rgb' in configs['Name']:
      num_input_channels = 3
      print('RGB_DIRECT_________')
    encoder = Encoder(Bottleneck, [3, 4, 6, 3],num_input_channels).to(device)
    encoder.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False).to(device)
    encoder.fc = nn.Linear(65536, configs['zsize']).to(device)
    encoder=encoder.to(device)

    #binary = Binary()

    decoder = Decoder().to(device)

    #direct_encoder = DirectEncoder(encoder).to(device)
    direct_encoder = Autoencoder(encoder).encoder.to(device)

    #from torchsummary import summary
    from torchinfo import summary
    print(type(direct_encoder))
    print(summary(direct_encoder,(1,num_input_channels,192,640)))

    return direct_encoder


def create_output_images(output_dir, model, dataloader, device, configs):
  model.eval()
  data_iter = iter(dataloader)
  images_original, optical_flow = next(data_iter)
  with torch.no_grad():
    print(images_original.shape)
    if configs['Name'] == 'VAR':
      images, latent_mu, latent_logvar = model(images_original.to(device))
    else:
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

def store_model(model_version, model, loss_values, configs,epoch, output_dir):
  if 'Same' in model_version:
    if loss_values['val_loss_avg'][-1] < loss_values['best_valid_loss_one']:
      loss_values['best_valid_loss_one'] = loss_values['val_loss_avg'][-1]
      torch.save(model.state_dict(), output_dir + 'ResNet18_'+ model_version + '_Dataset.pt')
      loss_values['best_epoch_same'] = epoch
  elif 'Other' in model_version:
    if loss_values['val_loss_avg_two'][-1] < loss_values['best_valid_loss_two']:
      loss_values['best_valid_loss_two'] = loss_values['val_loss_avg_two'][-1]
      torch.save(model.state_dict(), output_dir + 'ResNet18_'+ model_version + '_Dataset.pt')
      loss_values['best_epoch_other'] = epoch
  elif 'Final' in model_version:
    torch.save(model.state_dict(), output_dir + 'ResNet18_' + model_version + '_' + str(configs['num_epochs']) + 'E_Dataset.pt')
  elif 'Always' in model_version:
    torch.save(model.state_dict(), output_dir + 'ResNet18_' + epoch + 'E_Dataset.pt')
  else:
    print('model ' + model_version + ' unknown!')

def create_dirs(dir_one, dir_two, dir_three):

  top_directory = dir_one.split('/')[0]

  print(top_directory)

  if not os.path.exists(top_directory):
    os.makedirs(top_directory)
  if not os.path.exists(dir_one):
    os.makedirs(dir_one)
  if not os.path.exists(dir_two):
    os.makedirs(dir_two)
  if not os.path.exists(dir_three):
    os.makedirs(dir_three)

def plot_data_and_save_to_png(file_name):
    # Lese den gesamten Inhalt der Datei ein
    with open(file_name, 'r') as file:
        data_text = file.read()

    # Verwende reguläre Ausdrücke, um die relevanten Daten zu extrahieren
    pattern = r'Epoch \[(\d+) / \d+\] average reconstruction error: ([\d.]+)   validation error one: ([\d.]+)   validation error two: ([\d.]+)'
    matches = re.findall(pattern, data_text)

    # Erstelle eine Liste der extrahierten Daten
    data_list = [(int(match[0]), float(match[1]), float(match[2]), float(match[3])) for match in matches]

    # Erstelle ein DataFrame aus den extrahierten Daten
    df = pd.DataFrame(data_list, columns=['Epoch', 'Reconstruction Error', 'Validation Error One', 'Validation Error Two'])

    # Erstelle das Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['Epoch'], df['Reconstruction Error'], label='Reconstruction Error', marker='o')
    plt.plot(df['Epoch'], df['Validation Error One'], label='Validation Error One', marker='o')
    plt.plot(df['Epoch'], df['Validation Error Two'], label='Validation Error Two', marker='o')

    # Beschriftungen und Legende hinzufügen
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Error vs. Epoch')
    plt.legend()

    # Bestimme den Ausgabepfad für die PNG-Datei im gleichen Ordner wie die Eingabedatei
    output_dir = os.path.dirname(file_name)
    output_file_name = os.path.join(output_dir, os.path.splitext(os.path.basename(file_name))[0] + '.png')
    
    # Speichere das Plot in der PNG-Datei
    plt.grid(True)
    plt.savefig(output_file_name)
    plt.close()