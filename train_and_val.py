import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from ResNet import vae_loss

variational_beta = 1_000


def train(train_dataloader, model, optimizer, device):
  num_batches = 0
  train_loss_avg = 0

  for images, segmentations in train_dataloader:


    images = images.to(device)
    segmentations = segmentations.to(device)

    images = images.float()
    segmentations = segmentations.float()

    #print('images.shape: ',images.shape)
    
    if model.name == 'VAR':
      # autoencoder reconstruction
      print("image")
      print(images[0][0][0])
      assert (images >= 0).all() and (images <= 1).all() and not torch.isnan(images).any(), "images has values out of range!"
      print("images.shape: ", images.shape)
      images_recon, latent_mu, latent_logvar = model(images)

      print("recon_image")
      print(images_recon[0][0][0])
      assert (images_recon >= 0).all() and (images_recon <= 1).all() and not torch.isnan(images_recon).any(), "images_recon has values out of range!"

      # reconstruction error
      loss = vae_loss(images_recon, segmentations, latent_mu, latent_logvar, variational_beta)

    #elif model.name = "DIRECT":
    #  images_recon = model(images)

    else:
      print("images.shape")
      print(images.shape)
      # autoencoder reconstruction
      images_recon = model(images)
      #print(images_recon.shape)
      #print(segmentations.shape)
    
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

      if model.name == 'VAR':
        print("image")
        print(images[0][0][0])
        assert (images >= 0).all() and (images <= 1).all() and not torch.isnan(images).any(), "val images has values out of range or NaN values!"

        # autoencoder reconstruction
        images_recon, latent_mu, latent_logvar = model(images)
        print("recon_image")
        print(images_recon[0][0][0])
        assert (images_recon >= 0).all() and (images_recon <= 1).all() and not torch.isnan(images_recon).any(), "val images_recon has values out of range or NaN values!"


        # reconstruction error
        loss_val = vae_loss(images_recon, segmentations, latent_mu, latent_logvar, variational_beta)


      else:
        image_batch_recon = model(images)

        # reconstruction error
        loss_val = F.mse_loss(image_batch_recon, segmentations)

      val_loss_avg += loss_val.item()
      num_test_batches += 1

  val_loss_avg /= num_test_batches

  return val_loss_avg