import utils,config,model
import matplotlib.pyplot as plt
import torch
from d2l import torch as d2l
net_G,net_D = model.GenerateModel()

utils.LoadCheckPoint("my_checkpoint_D.pth.tar",net_D)
utils.LoadCheckPoint("my_checkpoint_G.pth.tar",net_G)
Z= torch.normal(0,1,(10,100,1,1))
prediction = net_G(Z)
prediction = prediction.detach().permute(0,2,3,1).numpy()
fig,axes = plt.subplots(2,5)
axes = axes.flatten()
for (ax,img) in zip(axes,prediction):
  ax.imshow(img)
  ax.axes.get_xaxis().set_visible(False)
  ax.axes.get_yaxis().set_visible(False)
fig.savefig("prediction.png")