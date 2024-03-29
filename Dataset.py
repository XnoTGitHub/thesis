from torch.utils.data import Dataset
from torchvision.io import read_image
import torch

import pandas as pd

class CarlaDataset(Dataset):
    def __init__(self, annotations_file, rgb_dir, seg_dir):
        self.img_labels = pd.read_csv('./' + annotations_file)
        self.rgb_dir = rgb_dir
        self.seg_dir = seg_dir
        #self.count = 0

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image_dir_suff = self.img_labels.iloc[idx, 0]
        #image_dir_suff = '../../shared/datasets/floriann/' + image_dir_suff
        #print(image_dir_suff)
        if self.seg_dir == 'direct':
          print("direct")
          if self.rgb_dir == 'rgb/':
            print("rgb")
            print(image_dir_suff)
            rgb_image = read_image('thesis/' + image_dir_suff)/255.
            steer = self.img_labels.iloc[idx, 1]
            #throttle = self.img_labels.iloc[idx, 2]
            label = torch.Tensor([steer]).float()
            seg_image = label#(steer,throttle)
          else:
            print("not rgb")
            index = image_dir_suff.find('rgb/')
            if index != -1:
              print(image_dir_suff)
              print(index)

              pref = image_dir_suff[:index]
              suff = image_dir_suff[index + len('rgb/'):]
              print(pref)
              print(suff)
              rgb_im_path = pref + self.rgb_dir + suff
              print(rgb_im_path)
              rgb_image = read_image('thesis/' + image_dir_suff)/255.
              steer = self.img_labels.iloc[idx, 1]
              throttle = self.img_labels.iloc[idx, 2]
              labels = torch.Tensor((steer,throttle)).float()#.to(device)
              seg_image = labels#(steer,throttle)


        elif self.rgb_dir in image_dir_suff:

          rgb_image = read_image('thesis/' + image_dir_suff)/255.
          #print(rgb_image.shape)

          index = image_dir_suff.find(self.rgb_dir)
          if index != -1:
            pref = image_dir_suff[:index]
            suff = image_dir_suff[index + len(self.rgb_dir):]
            seg_im_path = pref + self.seg_dir + suff
            #print(seg_im_path)
            #print(self.count)
            #self.cound += 1

            seg_image = read_image('thesis/' + seg_im_path)/255.

          else:
            print('ERROR, Label image not found: ',image_dir_suff)

        elif 'rgb/' in image_dir_suff:

          #rgb_image = read_image('./configs/' + image_dir_suff)/255.
          #print(rgb_image.shape)

          index = image_dir_suff.find('rgb/')
          if index != -1:
            print(image_dir_suff)
            print(index)

            pref = image_dir_suff[:index]
            suff = image_dir_suff[index + len('rgb/'):]
            print(pref)
            print(suff)
            rgb_im_path = pref + self.rgb_dir + suff
            seg_im_path = pref + self.seg_dir + suff
            print(seg_im_path)
            print(self.rgb_dir)
            print(self.seg_dir)
            #print(self.count)
            #self.cound += 1
            rgb_image = read_image('./configs/' + rgb_im_path[2:])/255.
            seg_image = read_image('./configs/' + seg_im_path[2:])/255.

          else:
            print('ERROR, Label image not found: ',image_dir_suff)

        else:
          print('ERROR, Input image not found: ',image_dir_suff)
        return rgb_image, seg_image
