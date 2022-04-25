import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

def train(train_dataloader, model, optimizer, device):
  num_batches = 0
  train_loss_avg = 0

  for images, segmentations in train_dataloader:


    images = images.to(device)
    segmentations = segmentations.to(device)

    images = images.float()
    segmentations = segmentations.float()
    
    # autoencoder reconstruction
    images_recon = model(images)
    
    # reconstruction error
    loss = F.mse_loss(images_recon, segmentations)
    
    # backpropagation
    optimizer.zero_grad()
    loss.backward()
    
    # one step of the optmizer (using the gradients from backpropagation)
    optimizer.step()
    
    train_loss_avg += loss.item()
    num_batches += 1
  return train_loss_avg / num_batches

def validate(valid_dataloader, model, optimizer, device):
  num_test_batches = 0
  val_loss_avg = 0
  for images, segmentations in  valid_dataloader:
      
    with torch.no_grad():

      images = images.to(device)
      segmentations = segmentations.to(device)

      images = images.float()
      segmentations = segmentations.float()

      image_batch_recon = model(images)

      # reconstruction error
      loss_val = F.mse_loss(image_batch_recon, segmentations)

      val_loss_avg += loss_val.item()
      num_test_batches += 1

  val_loss_avg /= num_test_batches

  return val_loss_avg