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

from utils import *
from my_logging import *
from train_and_val import *


import wandb
wandb.login(key='1a1619af1642c885007c0113e826c69a29d51689')

sweep_config = {
    'method': 'random'
    }
early_terminate = {
    'type': 'hyperband',
    'min_iter': 3
    }
sweep_config['early_terminate'] = early_terminate

metric = {
    'name': 'loss',
    'goal': 'minimize'   
    }

sweep_config['metric'] = metric

parameters_dict = {
#     'batch_size':{
#           'values': [32,64,128,256]
#         },
#      'learning_rate':{
#            'values': [1e-2,1e-3,1e-4]
#         },
#      'weight_decay':{
#            'values': [1e-4,1e-5,1e-6]
#         },
     'batch_size':{
           'values': [16,32,64,128,256]
         },
      'learning_rate':{
            'values': [1e-3]
         },
      'weight_decay':{
            'values': [1e-6]
         },

    }
sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="batch_lr_weightdec_120 epochs")

def run_model(config=None):
  with wandb.init(config=config):#, name = f"experiment_{wandb.config.learning_rate}_{wandb.config.weight_decay}_{wandb.config.batch_size}"):
    #config = wandb.config
    #wandb.init( project='Autoencoder Depth',
    #            name = f"experiment_{run}", 
    #            config={"batch_size": [32,64,128,256]})

    loss_values = get_loss_values()

    parser = argparse.ArgumentParser(description="Autoencoder")
    parser.add_argument("--config", help="YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
      configs = yaml.load(f,Loader=loader)

    ###################
    #### DataLoaer ####

    train_dataloader, valid_dataloader, valid_dataloader_two = get_dataloaders(configs['Name'], configs['TRAIN_SET'], configs['VAL_SET'], configs['VAL_SET_TWO'], wandb.config.batch_size)

    ################
    #### Model #####

    device = torch.device("cuda:0" if (torch.cuda.is_available() and configs['use_gpu']) else "cpu")

    if ('DEPTH' in configs['Name']) or ('VAR' in configs['Name']):
      autoencoder = create_model(configs, device)########sollte hier num_input_channels gleich 1 sein??!
      print(type(autoencoder))
    elif configs['Name'] == 'RGB':
      autoencoder = create_model(configs, device, num_input_channels=3)
    print(configs['Name'])
    num_params = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
    print('Number of parameters: %d' % num_params)

    TRAIN = configs['TRAIN']
    if (TRAIN):
      print('Training has started..')
      log_header(configs,autoencoder)

      optimizer = torch.optim.Adam(params=autoencoder.parameters() , lr=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)

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

          wandb.log({"train": loss_values['train_loss_avg'][-1],"val same": loss_values['val_loss_avg'][-1],"val other": loss_values['val_loss_avg_two'][-1]})

          log_epoch(configs, loss_values['train_loss_avg'][-1], loss_values['val_loss_avg'][-1], loss_values['val_loss_avg_two'][-1], epoch)
          store_model('Final_'+ configs['Name'], autoencoder, loss_values, configs, epoch)
      notifyTrainFinish('DEPTH to OPT_FLOW autoencoder')
    else:
      #if configs['Name'] == 'VAR':
      autoencoder.load_state_dict(torch.load('ResNet18_Final_' + configs['Name'] + '_' + str(configs['num_epochs']) + 'E_Dataset.pt'))
      #else:
      #  autoencoder.load_state_dict(torch.load('ResNet18_Final_' + configs['Name'] + 'E_Dataset.pt'))

    wandb.finish()

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
wandb.agent(sweep_id, run_model, count=10)
print('finish')