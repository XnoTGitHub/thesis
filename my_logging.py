import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchsummary import summary

from datetime import date, datetime

def log_header(configs, model):
  with open(configs['log_file'], "a") as myfile:
    myfile.write('\n')
    myfile.write('####################################\n')
    myfile.write('####### TRAINING Double Var ########\n')
    myfile.write('####################################\n')
    myfile.write('latent_dims %d\n' % (configs['latent_dims']))
    myfile.write('num_epochs %d\n'% (configs['num_epochs']))
    myfile.write('batch_size %d\n'% (configs['batch_size']))
    myfile.write('capacity %d\n'% (configs['capacity']))
    myfile.write('learning_rate %d\n'% (configs['learning_rate']))
    myfile.write('zsize %d\n' % (configs['zsize']))
    myfile.write('TRAIN_SET ' + configs['TRAIN_SET'] + '\n')
    myfile.write('VAL_SET ' + configs['VAL_SET'] + '\n')
    myfile.write('VAL_SET_TWO ' + configs['VAL_SET_TWO'] + '\n')
    myfile.write('Date ' + str(datetime.now()) + '\n')
    if configs['Name'] == 'DEPTH':
      myfile.write(str(summary(model,(1,192,640))))
    elif configs['Name'] == 'RGB':
      myfile.write(str(summary(model,(3,192,640))))
    elif 'DIRECT_rgb' in configs['Name']:
      myfile.write(str(summary(model,(3,192,640))))
    elif 'DIRECT' in configs['Name']:
      myfile.write(str(summary(model,(1,192,640))))
    myfile.write('\n')
    myfile.write('Train...\n')

def log_best(configs, best_epoch_same, best_epoch_other):
  with open(configs['log_file'], "a") as myfile:
    myfile.write('Best Epoch Same [%d] Best Epoch Other[%d] \n' % (best_epoch_same, best_epoch_other))

def log_epoch(configs, train_loss_avg, val_loss_avg, val_loss_avg_two, epoch):
  with open(configs['log_file'], "a") as myfile:
    myfile.write('Epoch [%d / %d] average reconstruction error: %f   validation error one: %f   validation error two: %f\n' % (epoch+1, configs['num_epochs'], train_loss_avg, val_loss_avg, val_loss_avg_two))
  print('Epoch [%d / %d] average reconstruction error: %f   validation error one: %f   validation error two: %f' % (epoch+1, configs['num_epochs'], train_loss_avg, val_loss_avg, val_loss_avg_two))
