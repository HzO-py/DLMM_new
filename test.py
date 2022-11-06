import torch
x=torch.tensor([[1,3,2,8],[5,1,3,2]],dtype=torch.float)
mean=x.mean(axis=0,keepdim=False)
inf=1e-5
std=torch.std(x,axis=0)+inf
y=torch.tensor([4,3,1,4],dtype=torch.float)
print(mean,std,(y-mean)/std)