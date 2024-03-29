import torch
from torch.autograd import Variable
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models,transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np 
import os
from torch.autograd import Function
from collections import OrderedDict
import torch.nn as nn
import math
zsize = 64
batch_size = 11
iterations =  500
learningRate= 0.000_1
from Action_Predicter_Models import Action_Predicter_LSTM

import torchvision.models as models
#ResNEt#####################################
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
    
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
###############################################################
class Double_Var_Encoder(nn.Module):

    def __init__(self, block, layers, num_classes=23):
        self.inplanes = 64
        super (Double_Var_Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#, return_indices = True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(5, stride=1)
        self.fc_mu = nn.Linear(512 * block.expansion, 1000)
        self.fc_logvar = nn.Linear(512 * block.expansion, 1000)
        self.fc_mu_two = nn.Linear(512 * block.expansion, 1000)
        self.fc_logvar_two = nn.Linear(512 * block.expansion, 1000)
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        #print('encoder start')
        x = self.conv1(x)
        #print(x.shape)
        x = self.bn1(x)
        #print(x.shape)
        x = self.relu(x)
        #print(x.shape)
        x = self.maxpool(x)
        #print('maxpool: ',x.shape)
        x = self.layer1(x)
        #print(x.shape)
        x = self.layer2(x)
        #print(x.shape)
        x = self.layer3(x)
        #print(x.shape)
        x = self.layer4(x)
        #print(x.shape)

        x = self.avgpool(x)
        #print('avgpool: ',x.shape)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        x_mu_two = self.fc_mu(x)
        x_logvar_two = self.fc_logvar(x)
        #print(x.shape)
        #print('end encoder')
        #return self.sigm(x_mu),self.sigm(x_logvar)
        return x_mu,x_logvar,x_mu_two,x_logvar_two



###############################################################
class Var_Encoder(nn.Module):

    def __init__(self, block, layers, num_classes=23):
        self.inplanes = 64
        super (Var_Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv1.weight.data.normal_(-0.001, 0.001)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#, return_indices = True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(5, stride=1)
        self.fc_mu = nn.Linear(512 * block.expansion, 1000)
        self.fc_logvar = nn.Linear(512 * block.expansion, 1000)
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        assert (x >= 0).all() and (x <= 1).all() and not torch.isnan(x).any(), "encoder start x images has values out of range!"

        x = self.conv1(x)

        x = self.relu(x)
        x = self.bn1(x)

        x = self.relu(x)

        x = self.maxpool(x)
        x = self.layer1(x)

        x = torch.sigmoid(x) #maybe change all the sigmoids to relu
        x = self.layer2(x)
        x = torch.sigmoid(x)
        x = self.layer3(x)
        x = torch.sigmoid(x)
        x = self.layer4(x)
        x = torch.sigmoid(x)

        x = self.avgpool(x)
        #print('avgpool: ',x.shape)
        x = x.view(x.size(0), -1)
        assert (x >= -1).all() and (x <= 1).all() and not torch.isnan(x).any(), "encoder x images has values out of range!"
        #print(x.shape)
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        #print(x.shape)
        #print('end encoder')
        #return self.sigm(x_mu),self.sigm(x_logvar)
        return x_mu,x_logvar


###############################################################
class Encoder(nn.Module):

    def __init__(self, block, layers, num_classes=23, num_input_channels=3):
        self.inplanes = 64
        super (Encoder, self).__init__()
        self.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#, return_indices = True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(5, stride=1)
        self.fc = nn.Linear(512 * block.expansion, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        #print('encoder start')
        x = self.conv1(x)
        #print(x.shape)
        x = self.bn1(x)
        #print(x.shape)
        x = self.relu(x)
        #print(x.shape)
        x = self.maxpool(x)
        #print('maxpool: ',x.shape)
        x = self.layer1(x)
        #print(x.shape)
        x = self.layer2(x)
        #print(x.shape)
        x = self.layer3(x)
        #print(x.shape)
        x = self.layer4(x)
        #print(x.shape)

        x = self.avgpool(x)
        #print('avgpool: ',x.shape)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.fc(x)
        #print(x.shape)
        #print('end encoder')
        return x
##########################################################################
class Binary(Function):

    @staticmethod
    def forward(ctx, input):
        return F.relu(Variable(input.sign())).data

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

##########################################################################
class Decoder(nn.Module):
  def __init__(self):
    super(Decoder,self).__init__()
    self.dfc3 = nn.Linear(zsize, 30720)
    self.bn3 = nn.BatchNorm2d(4096)
    self.dfc2 = nn.Linear(4096, 4096)
    self.bn2 = nn.BatchNorm2d(4096)
    self.dfc1 = nn.Linear(4096,256 * 6 * 6)
    self.bn1 = nn.BatchNorm2d(256*6*6)
    self.upsample1=nn.Upsample(scale_factor=2)
    self.dconv5 = nn.ConvTranspose2d(256, 256, 3, padding = 1)
    self.dconv4 = nn.ConvTranspose2d(256, 384, 3, padding = 1)
    self.dconv3 = nn.ConvTranspose2d(384, 192, 3, padding = 1)
    self.dconv2 = nn.ConvTranspose2d(192, 64, 5, padding = 2)
    self.dconv1 = nn.ConvTranspose2d(64, 3, 10, stride = 4, padding = 3)

  def forward(self,x):

    x = torch.sigmoid(x)
    x = self.dfc3(x)
    print(x)
    print(x.shape)
    x = F.relu(x)
    #assert (x >= 0).all() and (x <= 1).all(), "decoder x has values out of range!"
    x = x.view(-1,256,6,20)
    x=self.upsample1(x)
    x = self.dconv5(x)
    x = F.relu(x)
    x = F.relu(self.dconv4(x))
    x = F.relu(self.dconv3(x))
    #assert (x >= 0).all() and (x <= 1).all(), "decoder conv3 has values out of range!"
    x=self.upsample1(x)  
    x = self.dconv2(x)    
    x = F.relu(x)
    x=self.upsample1(x)
    x = self.dconv1(x)
    x = F.sigmoid(x)
    print('x:',torch.max(x))
    assert (x >= 0).all() and (x <= 1).all(), "decoder end x has values out of range!"
    return x

##########################################
class Autoencoder(nn.Module):
  def __init__(self,encoder):
    super(Autoencoder,self).__init__()
    self.name = 'none VAR'
    self.encoder = encoder
    self.binary = Binary()
    self.decoder = Decoder()

  def forward(self,x):
    #x=Encoder(x)
    x = self.encoder(x)
    print('Hier')
    print(x.shape)
    print(x)
#    print('encoder finish')
#    print(x.shape)
    #x = binary.apply(x)
    #print x
    #x,i2,i1 = self.binary(x)
    #x=Variable(x)
    x = self.decoder(x)
    return x
##########################################
class DirectEncoder(nn.Module): # not in use
  def __init__(self,encoder):
    super(DirectEncoder,self).__init__()
    self.name = 'none VAR'
    self.encoder = encoder
    self.binary = Binary()
    self.lstm = Action_Predicter_LSTM()

  def forward(self,x):
    #x=Encoder(x)
    x = self.encoder(x)
    #print(x.shape)
    #print(x)
#    print('encoder finish')
#    print(x.shape)
    #x = binary.apply(x)
    #print x
    #x,i2,i1 = self.binary(x)
    #x=Variable(x)
    x = self.lstm(x)
    return x

##########################################
class Var_Autoencoder(nn.Module):
  def __init__(self,encoder):
    super(Var_Autoencoder,self).__init__()
    self.name = 'VAR'
    self.encoder = encoder#.cpu()
    self.binary = Binary()
    self.decoder = Decoder()#.cpu()

  def forward(self,x):

    latent_mu, latent_logvar = self.encoder(x)
    #assert (latent_mu >= 0).all() and (latent_mu <= 1).all() and not torch.isnan(latent_mu).any(), "vor decoder latent_mu has values out of range!"
    #assert (latent_logvar >= 0).all() and (latent_logvar <= 1).all() and not torch.isnan(latent_logvar).any(), "vor decoder latent_logvar has values out of range!"
    print(latent_mu)
    print(latent_logvar)

    #latent = self.latent_sample(latent_mu, latent_logvar)
    latent = self.reparameterize(latent_mu,latent_logvar)
    print('latent.shape: ',latent.shape)
    #assert (latent >= 0).all() and (latent <= 1).all() and not torch.isnan(latent).any(), "vor decoder has values out of range!"
    x_recon = self.decoder(latent)

    return x_recon, latent_mu, latent_logvar

  def reparameterize(self, mu, logvar):
    """
    Reparameterization trick to sample from N(mu, var) from
    N(0,1).
    :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    :return: (Tensor) [B x D]
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu

  def latent_sample(self, mu, logvar):
    if self.training:
      # the reparameterization trick
      #print(mu,logvar)
      std = logvar.mul(0.5).exp_()
      eps = torch.empty_like(std).normal_()
      output = eps.mul(std).add_(mu)
      #print('max latent sample output: ', torch.max(output))
      return output
    else:
      print('validation')
      return mu


##########################################
class Double_Var_Autoencoder(nn.Module):
  def __init__(self,encoder):
    super(Double_Var_Autoencoder,self).__init__()
    self.name = 'VAR'
    self.encoder = encoder#.cpu()
    self.binary = Binary()
    self.decoder = Decoder()#.cpu()

  def forward(self,x):

    latent_mu_one, latent_logvar_one, latent_mu_two, latent_logvar_two= self.encoder(x)
    #print(latent_mu)
    #print(latent_logvar)

    #latent = self.latent_sample(latent_mu, latent_logvar)
    latent = self.reparameterize(latent_mu_one,latent_logvar_one,latent_mu_two,latent_logvar_two)
    print('latent.shape: ',latent.shape)
    x_recon = self.decoder(latent)

    return x_recon, latent_mu, latent_logvar

  def reparameterize(self, mu, logvar, mu_two, logvar_two):
    """
    Reparameterization trick to sample from N(mu, var) from
    N(0,1).
    :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    :return: (Tensor) [B x D]
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu

  def latent_sample(self, mu, logvar):
    if self.training:
      # the reparameterization trick
      #print(mu,logvar)
      std = logvar.mul(0.5).exp_()
      eps = torch.empty_like(std).normal_()
      output = eps.mul(std).add_(mu)
      #print('max latent sample output: ', torch.max(output))
      return output
    else:
      print('validation')
      return mu

def vae_loss(recon_x, x, mu, logvar,variational_beta):
    # recon_x is the probability of a multivariate Bernoulli distribution p.
    # -log(p(x)) is then the pixel-wise binary cross-entropy.
    # Averaging or not averaging the binary cross-entropy over all pixels here
    # is a subtle detail with big effect on training, since it changes the weight
    # we need to pick for the other loss term by several orders of magnitude.
    # Not averaging is the direct implementation of the negative log likelihood,
    # but averaging makes the weight of the other loss term independent of the image resolution.
    #recon_loss = F.binary_cross_entropy(recon_x.view(-1, 3*40960), x.view(-1, 3*40960), reduction='sum')
    print("recon_x.shape: ", recon_x[0][0][0])
    print("x.shape: ", x[0][0][0])

    assert (x >= 0).all() and (x <= 1).all() and not torch.isnan(x).any(), "x has values out of range or NaN values!"
    assert (recon_x >= 0).all() and (recon_x <= 1).all() and not torch.isnan(recon_x).any(), "recon_x has values out of range or NaN values!"

    #recon_loss = F.mse_loss(recon_x,x)
    recon_loss = F.binary_cross_entropy(recon_x.reshape(-1, 3*192*640), x.reshape(-1, 3*192*640), reduction='sum')
    
    # KL-divergence between the prior distribution over latent vectors
    # (the one we are going to sample from when generating new images)
    # and the distribution estimated by the generator for the given image.
    print('recon_loss: ', recon_loss)
    #kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
    print('kldivergence: ', kld_loss)
    
    return recon_loss + variational_beta * kld_loss

