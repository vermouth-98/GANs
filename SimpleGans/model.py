import torch
from torch import nn 
import config
class GeneratorBlock(nn.Module):
  """the Generator Module"""
  def __init__(self,out_channels,in_channels=3,kernel_size = 4,stride = 2,padding = 1, **kwargs):
    super(GeneratorBlock, self).__init__(**kwargs)
    self.conv2d_trans = nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride = stride,padding = padding,bias = False)
    self.bn = nn.BatchNorm2d(out_channels)
    self.activation = nn.ReLU()
  
  def forward(self,X):
    return self.activation(self.bn(self.conv2d_trans(X)))
class DiscriminatorBlock(nn.Module):
  """The Discriminator Module"""
  def __init__(self,out_channels,in_channels =3,kernel_size = 4,stride = 2, padding = 1,alpha = 0.2,**kwargs):
    super(DiscriminatorBlock,self).__init__(**kwargs)
    self.conv2d = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias = False)
    self.bn = nn.BatchNorm2d(out_channels)
    self.activation = nn.LeakyReLU(alpha,inplace=True)

  def forward(self,X):
    return self.activation(self.bn(self.conv2d(X)))

def GenerateModel():
  n_G = 64
  net_G = nn.Sequential(
    GeneratorBlock(in_channels = config.IN_CHANNELS_G,out_channels = n_G*8,stride = 1, padding = 0 ),
    GeneratorBlock(in_channels = n_G*8,out_channels = n_G*4 ),
    GeneratorBlock(in_channels = n_G*4,out_channels = n_G*2 ),
    GeneratorBlock(in_channels = n_G*2,out_channels = n_G ),
    nn.ConvTranspose2d(n_G,3,kernel_size = 4,stride = 2, padding = 1,bias = False),
    nn.Tanh()
  )
  n_D = 64
  net_D = nn.Sequential(
    DiscriminatorBlock(out_channels = n_D),
    DiscriminatorBlock(in_channels = n_D,out_channels = n_D*2),
    DiscriminatorBlock(in_channels = n_D*2 ,out_channels = n_D*4),
    DiscriminatorBlock(in_channels = n_D*4,out_channels = n_D*8),
    nn.Conv2d(n_D*8,1,kernel_size = 4,bias = False)
  )
  return net_G,net_D

if __name__ == '__main__':
  X= torch.rand((256,3,64,64))
  net_G,net_D = GenerateModel()
  Z = torch.normal(0,1,(256,100,1,1))
  fake_X = net_G(Z)
  fake_Y = net_D(fake_X.detach())
  real_Y= net_D(X)
  one=  torch.ones((config.BATCH_SIZE,))

  print(fake_X.shape,fake_Y.shape,real_Y.shape,one.reshape(real_Y.shape).shape)