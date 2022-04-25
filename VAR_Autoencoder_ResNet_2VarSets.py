import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image

import numpy as np
import csv
import pandas as pd

from Dataset import CarlaDataset
from ResNet import Encoder, Var_Encoder, Binary, Decoder, Autoencoder, Var_Autoencoder, Bottleneck, vae_loss

from datetime import date, datetime

latent_dims =64
num_epochs = 50
batch_size = 100
capacity = 64
learning_rate = 1e-5
zsize = 64
variational_beta = 1_000#1#100#10_000

TRAIN_SET = 'sets/Large_No_Traficlights.csv'
VAL_SET = 'sets/Small_No_Traficlights.csv'
VAL_SET_TWO = 'sets/7_No_Traficlights.csv'

#### Telegram ####
import requests

bot_token = '5012425653:AAHx4O6mzJCv90wundw7F7BKBBWSWgpRxgs'
bot_chatID = '313580257'

def tg_sendMessage(msg):
    assert(isinstance(msg, str))
    send_url = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + msg

    response = requests.get(send_url)

    return response.json

def notifyTrainFinish(name):
    assert(isinstance(name, str))
    tg_sendMessage('Training finished for '+name)

###################
#### DataLoader ####

dataset_train = CarlaDataset(TRAIN_SET, 'pred_depth/', 'opt_flow/')
dataset_val = CarlaDataset(VAL_SET, 'pred_depth/', 'opt_flow/')
dataset_val_two = CarlaDataset(VAL_SET_TWO, 'pred_depth/', 'opt_flow/')

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

train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, sampler=train_sampler)
valid_dataloader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, sampler=valid_sampler)
valid_dataloader_two = torch.utils.data.DataLoader(dataset_val_two, batch_size=batch_size, sampler=valid_sampler_two)


################
#### Model #####

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print('device: ', device)
encoder = Var_Encoder(Bottleneck, [3, 4, 6, 3]).to(device)
#encoder = Encoder(Bottleneck, [3, 4, 6, 3]).to(device)
encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False).to(device)
encoder.fc_mu = nn.Linear(65536, zsize).to(device)
encoder.fc_logvar = nn.Linear(65536, zsize).to(device)
encoder.fc_logvar.weight.data.normal_(-0.0001, 0.0001)#fill_(0.0001)
#encoder.fc = nn.Linear(65536, zsize).to(device)
encoder=encoder.to(device)

#binary = Binary()

decoder = Decoder().to(device)

autoencoder = Var_Autoencoder(encoder).to(device)

from torchsummary import summary

print(summary(autoencoder,(1,192,640)))


#print(type(model_depth['encoder'].parameters()))
num_params = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
print('Number of parameters: %d' % num_params)
best_valid_loss_one = float('inf')
best_valid_loss_two = float('inf')

TRAIN = True
if (TRAIN):
  with open("terminal_output.txt", "a") as myfile:
    myfile.write('\n')
    myfile.write('############################################################\n')
    myfile.write('####### TRAINING Variational Autoencoder Double Var ########\n')
    myfile.write('############################################################\n')
    myfile.write('latent_dims %d\n' % (latent_dims))
    myfile.write('num_epochs %d\n'% (num_epochs))
    myfile.write('batch_size %d\n'% (batch_size))
    myfile.write('capacity %d\n'% (capacity))
    myfile.write('learning_rate %d\n'% (learning_rate))
    myfile.write('zsize %d\n' % (zsize))
    myfile.write('TRAIN_SET ' + TRAIN_SET + '\n')
    myfile.write('VAL_SET ' + VAL_SET + '\n')
    myfile.write('VAL_SET_TWO ' + VAL_SET_TWO + '\n')
    myfile.write('Date ' + str(datetime.now()) + '\n')
    myfile.write('variational_beta ' + str(variational_beta) + '\n')
    myfile.write(str(summary(autoencoder,(1,192,640))))
    myfile.write('\n')
    myfile.write('Train...\n')

  optimizer = torch.optim.Adam(params=autoencoder.parameters() , lr=learning_rate, weight_decay=1e-5)

  # set to training mode
  autoencoder.train()

  train_loss_avg = []
  val_loss_avg = []
  val_loss_avg_two = []

  best_epoch_same = 0
  best_epoch_other = 0

  print('Training ...')
  for epoch in range(num_epochs):
      train_loss_avg.append(0)
      val_loss_avg.append(0)
      val_loss_avg_two.append(0)
      num_batches = 0
      
      for images, segmentations in train_dataloader:

          images = images.to(device)
          segmentations = segmentations.to(device)

          images = images.float()
          segmentations = segmentations.float()

          #print(autoencoder.device)
          autoencoder = autoencoder.to(device)
          # autoencoder reconstruction
          images_recon, latent_mu, latent_logvar = autoencoder(images)
          print('image_recon: ',torch.max(images_recon))
          
          # reconstruction error
          #loss = F.mse_loss(images_recon, segmentations)
          loss = vae_loss(images_recon, segmentations, latent_mu, latent_logvar, variational_beta)
          
          # backpropagation
          optimizer.zero_grad()
          loss.backward()
          
          # one step of the optmizer (using the gradients from backpropagation)
          optimizer.step()
          
          train_loss_avg[-1] += loss.item()
          num_batches += 1

      train_loss_avg[-1] /= num_batches
      num_test_batches = 0
      for images, segmentations in  valid_dataloader:
          
          with torch.no_grad():

              images = images.to(device)
              segmentations = segmentations.to(device)

              images = images.float()
              segmentations = segmentations.float()

              images_recon, latent_mu, latent_logvar = autoencoder(images)

              # reconstruction error
              #loss_val = F.mse_loss(image_batch_recon, segmentations)
              loss_val = vae_loss(images_recon, segmentations, latent_mu, latent_logvar, variational_beta)

              val_loss_avg[-1] += loss_val.item()
              num_test_batches += 1

      val_loss_avg[-1] /= num_test_batches
      if val_loss_avg[-1] < best_valid_loss_one:
        best_valid_loss_one = val_loss_avg[-1]
        best_epoch_same = epoch
        torch.save(autoencoder.state_dict(),'Var_ResNet18_Same_Dataset.pt')
      num_test_batches = 0
      #Validation Two
      for images, segmentations in  valid_dataloader_two:
          
          with torch.no_grad():

              images = images.to(device)
              segmentations = segmentations.to(device)

              images = images.float()
              segmentations = segmentations.float()

              images_recon, latent_mu, latent_logvar = autoencoder(images)

              # reconstruction error
              #loss_val = F.mse_loss(image_batch_recon, segmentations)
              loss_val = vae_loss(images_recon, segmentations, latent_mu, latent_logvar, variational_beta)

              val_loss_avg_two[-1] += loss_val.item()
              num_test_batches += 1

      val_loss_avg_two[-1] /= num_test_batches
      if val_loss_avg_two[-1] < best_valid_loss_two:
        best_valid_loss_two = val_loss_avg_two[-1]
        best_epoch_other = epoch
        torch.save(autoencoder.state_dict(),'Var_ResNet18_Other_Dataset.pt')
      with open("terminal_output.txt", "a") as myfile:
        myfile.write('Epoch [%d / %d] average reconstruction error: %f   validation error Same: %f   validation error Other: %f\n' % (epoch+1, num_epochs, train_loss_avg[-1], val_loss_avg[-1], val_loss_avg_two[-1]))
      print('Epoch [%d / %d] average reconstruction error: %f   validation error Same: %f   validation error Other: %f' % (epoch+1, num_epochs, train_loss_avg[-1], val_loss_avg[-1], val_loss_avg_two[-1]))
  torch.save(autoencoder.state_dict(),'Var_ResNet18_' + str(num_epochs) + 'E_Dataset.pt')
  notifyTrainFinish('DEPTH to OPT_FLOW autoencoder')
else:
  #model.load_state_dict(torch.load('model_state.pth'))
  autoencoder.load_state_dict(torch.load('Var_ResNet18_' + str(num_epochs) + 'E_Dataset.pt'))

# Dataset Final
autoencoder.eval()
images_original, optical_flow = iter(valid_dataloader).next()
with torch.no_grad():
  print(images_original.shape)
  #images = autoencoder(images_original.to(device))
  images, _, _ = autoencoder(images_original.to(device))
  images = images.cpu()
  print(images.shape)
  for i in range(0,images.shape[0]):
    output_tensor = torch.zeros(3,192*3,640)
    image_tensor = images[i,:,:,:]
    #im = transforms.ToPILImage()(image_tensor).convert("RGB")
    #im.save("autoencoder_output/image_" + str(i) + ".png")
    label_tensor = optical_flow[i,:,:,:]
    #label = transforms.ToPILImage()(label_tensor).convert("RGB")
    #label.save("autoencoder_output/label_" + str(i) + ".png")
    depth_tensor = images_original[i,:,:,:]
    #depth = transforms.ToPILImage()(depth_tensor).convert("RGB")
    #depth.save("autoencoder_output/depth_" + str(i) + ".png")
    #images (batch_size, 3, 192, 640)

    output_tensor[:,:192,:] = images[i,:,:,:]
    output_tensor[:,192:2*192,:] = optical_flow[i,:,:,:]
    output_tensor[:,2*192:,:] = images_original[i,:,:,:]
    output = transforms.ToPILImage()(output_tensor).convert("RGB")
    output.save("autoencoder_output_final/frame_" + str(i) + ".png")

# Dataset Same
autoencoder.load_state_dict(torch.load('Var_ResNet18_Same_Dataset.pt'))

autoencoder.eval()
images_original, optical_flow = iter(valid_dataloader).next()
with torch.no_grad():
  print(images_original.shape)
  #images = autoencoder(images_original.to(device))
  images, _, _ = autoencoder(images_original.to(device))
  images = images.cpu()
  print(images.shape)
  for i in range(0,images.shape[0]):
    output_tensor = torch.zeros(3,192*3,640)
    image_tensor = images[i,:,:,:]
    #im = transforms.ToPILImage()(image_tensor).convert("RGB")
    #im.save("autoencoder_output/image_" + str(i) + ".png")
    label_tensor = optical_flow[i,:,:,:]
    #label = transforms.ToPILImage()(label_tensor).convert("RGB")
    #label.save("autoencoder_output/label_" + str(i) + ".png")
    depth_tensor = images_original[i,:,:,:]
    #depth = transforms.ToPILImage()(depth_tensor).convert("RGB")
    #depth.save("autoencoder_output/depth_" + str(i) + ".png")
    #images (batch_size, 3, 192, 640)

    output_tensor[:,:192,:] = images[i,:,:,:]
    output_tensor[:,192:2*192,:] = optical_flow[i,:,:,:]
    output_tensor[:,2*192:,:] = images_original[i,:,:,:]
    output = transforms.ToPILImage()(output_tensor).convert("RGB")
    output.save("autoencoder_output_same/frame_" + str(i) + ".png")

# Dataset Other
autoencoder.load_state_dict(torch.load('Var_ResNet18_Other_Dataset.pt'))


autoencoder.eval()
images_original, optical_flow = iter(valid_dataloader).next()
with torch.no_grad():
  print(images_original.shape)
  #images = autoencoder(images_original.to(device))
  images, _, _ = autoencoder(images_original.to(device))
  images = images.cpu()
  print(images.shape)
  for i in range(0,images.shape[0]):
    output_tensor = torch.zeros(3,192*3,640)
    image_tensor = images[i,:,:,:]
    #im = transforms.ToPILImage()(image_tensor).convert("RGB")
    #im.save("autoencoder_output/image_" + str(i) + ".png")
    label_tensor = optical_flow[i,:,:,:]
    #label = transforms.ToPILImage()(label_tensor).convert("RGB")
    #label.save("autoencoder_output/label_" + str(i) + ".png")
    depth_tensor = images_original[i,:,:,:]
    #depth = transforms.ToPILImage()(depth_tensor).convert("RGB")
    #depth.save("autoencoder_output/depth_" + str(i) + ".png")
    #images (batch_size, 3, 192, 640)

    output_tensor[:,:192,:] = images[i,:,:,:]
    output_tensor[:,192:2*192,:] = optical_flow[i,:,:,:]
    output_tensor[:,2*192:,:] = images_original[i,:,:,:]
    output = transforms.ToPILImage()(output_tensor).convert("RGB")
    output.save("autoencoder_output_other/frame_" + str(i) + ".png")

if (TRAIN):
  with open("terminal_output.txt", "a") as myfile:
    myfile.write('Best Epoch Same [%d] Best Epoch Other[%d] \n' % (best_epoch_same, best_epoch_other))

print('finish')
