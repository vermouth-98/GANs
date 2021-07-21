import torchvision
import torch
BATCH_SIZE= 256
transformer = torchvision.transforms.Compose([
  torchvision.transforms.Resize((64,64)),
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(0.5,0.5),
])
DATA_DIR  = "pokemon"
NUM_WORKERS = 2
PIN_MEMORY = True
IN_CHANNELS_G = 100
NUM_EPOCHS = 20
LEARNING_RATE = 0.005
LATTENT_DIM=100
DEVICE = torch.device('cuda:0')
BETAS = [0.5,0.999]
LOAD_NET = True
SAVE_NET= True
CHECKPOINT_PATH = ""