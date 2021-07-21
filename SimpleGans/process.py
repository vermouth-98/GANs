import torch
import config,utils
from tqdm.autonotebook import tqdm 
from torch import nn
from d2l import torch as d2l

def train(net_D, net_G, data_iter):
  net_D.train()
  net_G.train()
  loss = nn.BCEWithLogitsLoss(reduction= "sum")
  optimizer_hp = {"lr": config.LEARNING_RATE,"betas": config.BETAS}
  optimizer_D = torch.optim.Adam(net_D.parameters(),**optimizer_hp)
  optimizer_G = torch.optim.Adam(net_G.parameters(),**optimizer_hp)
  if config.LOAD_NET:
    utils.LoadCheckPoint(config.CHECKPOINT_PATH+ "my_checkpoint_D.pth.tar",net_D,optimizer_D)
    utils.LoadCheckPoint(config.CHECKPOINT_PATH + "my_checkpoint_G.pth.tar",net_G,optimizer_G)
  else:
    for w in net_D.parameters():
      nn.init.normal_(w,0,0.02)
    for w in net_G.parameters():
      nn.init.normal_(w,0,0.02)
  
  Metric = utils.Metric(3)
  for epoch in range(config.NUM_EPOCHS):
    Metric.reset(3)
    loop= tqdm(data_iter,leave = True)
    for i,(X,_) in enumerate(loop): 
       
      X= X.to(config.DEVICE)
      batch_size= X.shape[0]
      Z= torch.normal(0,1,size = (batch_size,config.LATTENT_DIM,1,1),device = config.DEVICE)
      Metric.add( utils.update_D(X,Z,net_D,net_G,loss,optimizer_D),
                  utils.update_G( Z, net_D, net_G, loss, optimizer_G),
                  batch_size)
    
    
    print("epoch: {}, loss discriminator: {}, loss generator: {}".format(epoch+1,Metric[0]/Metric[2],Metric[1]/Metric[2]))
  if config.SAVE_NET:
    utils.SaveCheckPoint(net_D,optimizer_D,config.CHECKPOINT_PATH+"my_checkpoint_D.pth.tar")
    utils.SaveCheckPoint(net_G,optimizer_G,config.config.CHECKPOINT_PATH+"my_checkpoint_G.pth.tar")

