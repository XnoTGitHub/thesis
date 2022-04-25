import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import csv
import pandas as pd
from shutil import copy

# I/O libraries
import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

# Helper libraries
import numpy as np
from PIL import Image
from PIL import ImageOps
import IPython

from Dataset import CarlaDataset
from ResNet import Encoder, Binary, Decoder, Autoencoder, Bottleneck

from torchvision.io import read_image
import torchvision.transforms as transforms

PATH_TO_DATA = 'bearcar/'
zsize = 64

####TELEGRAM####
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
###############

use_gpu = True

from models.network.encoder import resnet_encoder
from models.network.rsu_decoder import RSUDecoder
from models.network.depth_decoder import DepthDecoder
model_depth = {}
device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

model_depth["encoder"] = resnet_encoder(num_layers=18, num_inputs=1,
                                        pretrained=True).to(device)

if True:
    model_depth["depth_decoder"] = RSUDecoder(num_output_channels=1, use_encoder_disp=True,
                                              encoder_layer_channels=model_depth["encoder"].layer_channels).to(device)
else:
    model_depth["depth_decoder"] = DepthDecoder(num_output_channels=1,
                                                encoder_layer_channels=model_depth["encoder"].layer_channels).to(device)

checkpoint = torch.load('models/model18_192x640.pth.tar',map_location=torch.device(device))
#model.load_state_dict(checkpoint)

for m_name, _ in model_depth.items():
  if m_name in checkpoint:
    model_depth[m_name].load_state_dict(checkpoint[m_name])
  else:
    print("There is no weight in checkpoint for model {}".format(m_name))







for m in model_depth.values():
    m.eval()

PROCESS = True

if PROCESS:

  for town in os.listdir(PATH_TO_DATA):

    PATH_TO_RGB = PATH_TO_DATA + town + '/rgb/'
    PATH_TO_DEPTH = PATH_TO_DATA + town + '/pred_depth/'

    if not os.path.isdir(PATH_TO_DEPTH):
      os.mkdir(PATH_TO_DEPTH)
    else:
      print(PATH_TO_DATA + town, ' already exists!')
      continue

    max = 0
    print(PATH_TO_DEPTH)
    for view_dir in os.listdir(PATH_TO_RGB):
      if not os.path.isdir(PATH_TO_DEPTH  + view_dir + '/'):
        os.mkdir(PATH_TO_DEPTH  + view_dir + '/')
      count = 0
      for filename in os.listdir(PATH_TO_RGB + view_dir + '/'):
        if filename[-4:] in '.png.jpg':
          print(count, filename)
          count +=1
          Path_to_rgb_image = PATH_TO_RGB + view_dir + '/' + filename
          if filename[-4:] in '.jpg':
            Path_to_seg_image = PATH_TO_DEPTH + view_dir + '/' + filename[:-3] + 'png'
          else:
            Path_to_seg_image = PATH_TO_DEPTH + view_dir + '/' + filename
          print(Path_to_seg_image)
          original_im = Image.open(Path_to_rgb_image)
          #original_im = original_im.resize((640,192), Image.ANTIALIAS)
          original_im = original_im.resize((640,360), Image.ANTIALIAS)
          width, height = original_im.size
          #print(width,height)
          original_im = original_im.crop((0, 58, 640, height - 110))
          #print(type(original_im))
          prep_im = np.transpose(original_im, (2, 0, 1 ))#change type to numpy
          prep_im = prep_im[None, ...]
          prep_im = torch.tensor(prep_im)
          prep_im = prep_im.float()/255.
          prep_im = prep_im.to(device)
          print(prep_im.shape)

          embedding = model_depth["encoder"](prep_im)
          #print(images[0].shape)
          seg_map = model_depth["depth_decoder"](embedding)
          #print(seg_map[0].shape)
          seg_numpy = seg_map[0].cpu().detach().numpy()
          #print(seg_numpy[0,0,:,:].max()*255)
          if seg_numpy[0,0,:,:].max() > max:
            max = seg_numpy[0,0,:,:].max()
          seg_image = Image.fromarray((seg_numpy[0,0,:,:]*255*6).astype(np.uint8))#####No Idea why I have to multiply it by 6!!!
          #seg_image = Image.fromarray(seg_numpy)
          #plt.imshow(original_im)
          #plt.show()
          #plt.imshow(seg_image)
          #plt.show()
          seg_image.save(Path_to_seg_image)
          
        elif filename[-4:] in '.csv':

          copy(PATH_TO_RGB + view_dir + '/' + filename, PATH_TO_DEPTH + view_dir + '/' + filename)

          #df = pd.read_csv(PATH_TO_RGB + dataset +  '/' + view_dir + '/' + filename).values.tolist()
          #print(df)
          #df = [[PATH_TO_SEG + dataset + '/' + name.split('/')[-1], thr, ste] for name,thr,ste in df]
          #print(df)
        else:
          print('filetype not known:',filename)
    print(max)  
    print('finish')
    tg_sendMessage("Depth Prediction Done: " + PATH_TO_DEPTH)



#######################
#### Optical Flow #####

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

encoder = Encoder(Bottleneck, [3, 4, 6, 3]).to(device)
encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False).to(device)
encoder.fc = nn.Linear(65536, zsize).to(device)
encoder=encoder.to(device)

#binary = Binary()

decoder = Decoder().to(device)

autoencoder = Autoencoder(encoder).to(device)

autoencoder.load_state_dict(torch.load('ResNet18_Other_Dataset.pt',map_location=torch.device(device)))

autoencoder.eval()

PROCESS_OPT = True

if PROCESS_OPT:

  for town in os.listdir(PATH_TO_DATA):

    PATH_TO_RGB = PATH_TO_DATA + town + '/pred_depth/'
    PATH_TO_DEPTH = PATH_TO_DATA + town + '/pred_opt_other/'

    if not os.path.isdir(PATH_TO_DEPTH):
      os.mkdir(PATH_TO_DEPTH)
    else:
      print(PATH_TO_DATA + town, ' already exists!')
      continue

    max = 0
    print(PATH_TO_DEPTH)
    for view_dir in os.listdir(PATH_TO_RGB):
      if not os.path.isdir(PATH_TO_DEPTH  + view_dir + '/'):
        os.mkdir(PATH_TO_DEPTH  + view_dir + '/')
      count = 0
      for filename in os.listdir(PATH_TO_RGB + view_dir + '/'):
        if filename[-4:] in '.png.jpg':
          print(count, filename)
          count +=1
          Path_to_rgb_image = PATH_TO_RGB + view_dir + '/' + filename
          Path_to_seg_image = PATH_TO_DEPTH + view_dir + '/' + filename
          print(Path_to_seg_image)
          original_im = Image.open(Path_to_rgb_image)
          #original_im = read_image(Path_to_rgb_image)
          original_im = original_im.resize((640,192), Image.ANTIALIAS)
          width, height = original_im.size
          print(width,height)
          #print(original_im.shape)
          #original_im = original_im.crop((0, 58, 640, height - 110))
          #original_im = original_im.resize((640,192), Image.ANTIALIAS)
          #print(type(original_im))
          #print('original_im.shape: ',original_im.shape)
          #images = [original_im]
          #prep_im = np.transpose(original_im, (2, 0, 1 ))#change type to numpy
          prep_im = np.transpose(original_im, (0, 1))#change type to numpy
          prep_im = prep_im[None,None, ...]
          prep_im = torch.tensor(prep_im)
          prep_im = prep_im.float()/255.
          prep_im = prep_im.to(device)
          print(prep_im.shape)

          seg_map = autoencoder(prep_im)
          print(seg_map.shape)
          seg_numpy = seg_map[0].cpu().detach().numpy()
          #print(seg_numpy[0,0,:,:].max()*255)
          if seg_numpy[0,:,:].max() > max:
            max = seg_numpy[0,:,:].max()
          #seg_image = Image.fromarray((seg_numpy[0,:,:]*255).astype(np.uint8))#####No Idea why I have to multiply it by 6!!!
          print('max: ',max)
          #seg_image = Image.fromarray(seg_numpy)
          #plt.imshow(original_im)
          #plt.show()
          #plt.imshow(seg_image)
          #plt.show()
          seg_numpy = np.transpose(seg_numpy, (1, 2, 0 ))
          output = transforms.ToPILImage()((seg_numpy*255).astype(np.uint8)).convert("RGB")
          output.save(Path_to_seg_image)
          
        elif filename[-4:] in '.csv':

          copy(PATH_TO_RGB + view_dir + '/' + filename, PATH_TO_DEPTH + view_dir + '/' + filename)

          #df = pd.read_csv(PATH_TO_RGB + dataset +  '/' + view_dir + '/' + filename).values.tolist()
          #print(df)
          #df = [[PATH_TO_SEG + dataset + '/' + name.split('/')[-1], thr, ste] for name,thr,ste in df]
          #print(df)
        else:
          print('filetype not known:',filename)
    print(max)  
    print('finish')
    tg_sendMessage("Opt Prediction Done: " + PATH_TO_DEPTH)

