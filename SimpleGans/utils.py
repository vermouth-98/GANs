import torch
import config
class Metric():
  def __init__(self,num):
    self.data = [0.]*num
  
  def reset(self,num):
    self.data = [0.]*num
  
  def add(self,*args):
    self.data =[a+float(b) for a, b in zip(self.data,args)]
  
  def __getitem__(self,idx):
    return self.data[idx]

def update_D(X,Z,net_D,net_G,loss,optimizer_D):
  optimizer_D.zero_grad()
  fake_X = net_G(Z)
  fake_Y = net_D(fake_X.detach())

  real_Y = net_D(X)
  one = torch.ones_like(real_Y,device = config.DEVICE)
  zero = torch.zeros_like(fake_Y,device = config.DEVICE)
  l = 0.5*(loss(real_Y,one)+loss(fake_Y,zero))
  l.backward()
  optimizer_D.step()
  return l

def update_G(Z,net_D,net_G,loss,optimizer_G):
  optimizer_G.zero_grad()
  fake_X= net_G(Z)
  fake_Y = net_D(fake_X)
  one = torch.ones_like(fake_Y,device = config.DEVICE)
  l = loss(fake_Y,one)
  l.backward()
  optimizer_G.step()
  return l

def SaveCheckPoint(net,optimizer,filename = "my_checkpoint.pth.tar"):
  print("=> saving checkpoint")
  checkpoint = {
    "state_dict": net.state_dict(),
    "optimizer" : optimizer.state_dict(),
  }
  torch.save(checkpoint,filename)
  print("done!")

def LoadCheckPoint(chechpoint_file,net,optimizer=None):
  print("=> loading checkpoint")
  checkpoint= torch.load(chechpoint_file,map_location= config.DEVICE)
  net.load_state_dict(checkpoint['state_dict'])
  if optimizer:
    optimizer.load_state_dict(checkpoint['optimizer'])

    for param_group in optimizer.param_group:
      param_group['lr'] = config.LEARNING_RATE
      param_group['betas'] = config.BETAS
  print("done!")