import torch
import numpy as np
from PIL import Image
from utils import getSample,readCsv,fileFeatureExtraction,getFaceSample,getBioSample,extractGsr,npyStandard
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import ToTensor, Resize, RandomCrop,Compose,RandomHorizontalFlip

class BioDataset(Dataset):

    def __init__(self,train,train_rio,modal):
        self.items=getBioSample('/hdd/lzq/data_2022.1.29/pain2/bio',"/hdd/lzq/data_2022.1.29/pain2/label.csv")
        self.train_rio=int(len(self.items)*train_rio)
        self.modal=modal
        if train:
            self.items=self.items[:self.train_rio]
        else:
            self.items=self.items[self.train_rio:]
        print(len(self.items))

    def __len__(self):
        return len(self.items)

    def load_seq(self,file_path):
        seq=readCsv(self.modal,file_path)
        # try:
        seq=extractGsr(np.array(seq))
        
        # except:
        #     raise Exception(seq)
        return seq

    def __getitem__(self, idr):
        item=self.items[idr]
        x=self.load_seq(item[0])
        x=torch.tensor(x,dtype=torch.float32)
        x=torch.flatten(x)
        sample = {'x': x,'y':int(item[-1])}
        return sample    
        
   
class FaceDataset(Dataset):
    def __init__(self,train,train_rio,paths):
        self.items=[]
        for path in paths:
            self.items+=getFaceSample(path[0],path[1],path[2])
        self.train_rio=int(len(self.items)*train_rio)
        if train:
            self.items=self.items[:self.train_rio]
        else:
            self.items=self.items[self.train_rio:]
        print(len(self.items))
        items=[]
        for person in self.items:
            for img in person:
                items.append(img)
        self.items=items
        print(len(self.items))
        self.transform = Compose([Resize(64),RandomCrop(48),RandomHorizontalFlip(),ToTensor()])

    def load_img(self,file_path):
        img = Image.open(file_path).convert('L')
        img=np.array(img)
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        img=self.transform(img)
        return img

    def load_npy(self,file_path):
        npy=np.load(file_path)
        npy=npyStandard(npy)
        return npy

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idr):
        item=self.items[idr]
        img=self.load_img(item[0])
        npy=self.load_npy(item[1])
        sample = {'x1': img,'x2':npy,'y':item[-1]}
        return sample

# dataset=BioDataset(1,0.9,"gsr")
# train_dataloader = DataLoader(dataset, batch_size=2,shuffle=True)
# for i,data in enumerate(train_dataloader):
#     x,y=data.values()
#     print(x,y)