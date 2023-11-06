import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.io import read_image

from PIL import Image

import numpy as np
import csv
import pandas as pd

from Dataset import CarlaDataset
from ResNet import Encoder, Binary, Decoder, Autoencoder, Bottleneck

batch_size = 10
zsize = 64

SET_DIR = 'sets/'

TRAIN_FILE = 'Large_No_Traficlights.csv'
VAL_FILE = 'Small_No_Traficlights.csv'
VAL_FILE_TWO = '7_No_Traficlights.csv'

TRAIN_SET = SET_DIR + TRAIN_FILE
VAL_SET = SET_DIR + VAL_FILE
VAL_SET_TWO = SET_DIR + VAL_FILE_TWO


###################
#### DataLoaer ####

dataset_train = CarlaDataset(TRAIN_SET, 'pred_depth/', 'ofl/')
dataset_val = CarlaDataset(VAL_SET, 'pred_depth/', 'ofl/')
dataset_val_two = CarlaDataset(VAL_SET_TWO, 'pred_depth/', 'ofl/')

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

encoder = Encoder(Bottleneck, [3, 4, 6, 3]).to(device)
encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False).to(device)
encoder.fc = nn.Linear(65536, zsize).to(device)
encoder=encoder.to(device)

decoder = Decoder().to(device)

autoencoder = Autoencoder(encoder).to(device)

autoencoder.load_state_dict(torch.load('ResNet18_Same_Dataset.pt'))

autoencoder.eval()

################
# Save Latents #

def process_latents(set_dir,csv_file_name, save=False,print_output=False):
	lines_with_latents = []

	file_to_read = set_dir + csv_file_name

	df = pd.read_csv(file_to_read)
	csv_entries = df.values.tolist()
	for idx, entry in enumerate(csv_entries):
		file_name, throttle, steer = entry
		image = read_image(file_name)/255.
		image = image.to(device)
		image = image.float()

		latent = autoencoder.encoder(image[None, ...]).cpu().detach().numpy()[0]
		#print(idx,latent)
		print(idx)
		lines_with_latents.append([file_name,throttle,steer,latent])

	if print_output:
		print(len(csv_entries))
		print(len(lines_with_latents))
		print(lines_with_latents[0])
		print(len(lines_with_latents[0]))
	if save:
		np_samples = np.asarray(lines_with_latents)
		output_file_name = 'latents/control_latents_ResNet18_' + csv_file_name[:-3] + 'npy'
		np.save(output_file_name,np_samples)



##################
###### MAIN ######

print('##########################')
print('##Process ' + TRAIN_SET + '##')
print()
process_latents(SET_DIR,TRAIN_FILE,save=True,print_output=True)
print()
print('##########################')
print('##Process ' + VAL_SET + '##')
print()
process_latents(SET_DIR,VAL_FILE,save=True,print_output=True)
print()
print('##########################')
print('##Process ' + VAL_SET_TWO + '##')
print()
process_latents(SET_DIR,VAL_FILE_TWO,save=True,print_output=True)
