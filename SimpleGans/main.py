import torch
import torchvision
import config,model,process
import warnings
def run():
  warnings.filterwarnings('ignore')
  dataset = torchvision.datasets.ImageFolder(config.DATA_DIR)
  dataset.transform = config.transformer
  data_iter = torch.utils.data.DataLoader(
    dataset,batch_size=config.BATCH_SIZE,
    shuffle= True,num_workers= config.NUM_WORKERS,drop_last=False, pin_memory=config.PIN_MEMORY
  )
  net_G,net_D = model.GenerateModel()
  net_G,net_D = net_G.to(config.DEVICE), net_D.to(config.DEVICE)
  
  
    

  process.train(net_D,net_G,data_iter)