import torch
import numpy as np
from PIL import Image
from utils import getSample,readCsv,fileFeatureExtraction,getFaceSample,getBioSample,extractGsr,npyStandard,getVoiceSample
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import ToTensor, Resize, RandomCrop,Compose,RandomHorizontalFlip,RandomVerticalFlip,ColorJitter
import random
import os
from models import Prototype,Classifier,ResNet18,cnn1d,VGG,Regressor

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
    def __init__(self,train,train_rio,paths,is_person,cls_threshold,r_threshold):
        self.items=[]
        for path in paths:
            self.items+=getFaceSample(path[0],path[1],path[2])
        self.train_rio=int(len(self.items)*train_rio)
        if train:
            self.items=self.items[:self.train_rio]
            self.transform = Compose([Resize([52,52]),RandomCrop([48,48]),RandomHorizontalFlip(),RandomVerticalFlip(),ColorJitter(0.1,0.1,0.1,0.1),ToTensor()])
        else:
            self.items=self.items[self.train_rio:]
            self.transform = Compose([Resize([52,52]),ToTensor()])

        items,items1,items2=[],[],[]
        if not is_person:
            for person in self.items:
                for img in person:
                    if cls_threshold is not None:
                        if float(img[-1])<=cls_threshold[0]:
                            img[-1]=int(0)
                            items1.append(img)
                        elif float(img[-1])>=cls_threshold[1]:
                            img[-1]=int(1)
                            items2.append(img)
                    else:
                        items.append(img)
            if cls_threshold is not None:
                if train:
                    if len(items1)>len(items2):
                        self.items=random.sample(items1,len(items2))+items2
                    else:
                        self.items=random.sample(items2,len(items1))+items1
                else:
                    self.items=items1+items2
            else:    
                self.items=items
        
        self.is_person=is_person
        self.r_threshold=r_threshold

        if cls_threshold is None:
            self.videoInit()

        print(len(self.items),self.r_threshold)

    def videoInit(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        net2 = VGG("VGG19")
        checkpoint = torch.load(os.path.join("/hdd/sdd/lzq/DLMM_new/model/logs/face_v3.2.t7"))
        print(checkpoint["acc"])
        net2.load_state_dict(checkpoint['net2'])
        net2 = net2.to(device)
        net3 = Classifier(512,64,2)
        net3.load_state_dict(checkpoint['net3'])
        net3 = net3.to(device)
        
        items=[]

        net2.eval()
        net3.eval()
        with torch.no_grad():
            for item in self.items:
                if item[-1]>self.r_threshold:

                    x=self.load_img(item[0]).unsqueeze(0).to(device)
                    x = net2(x)
                    outputs = net3(x)
                    _, predicted = torch.max(outputs.data, 1)
                    predicted=int(predicted.cpu())

                    if predicted==0:
                        continue
                
                items.append(item)
        self.items=items

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
        imgs=[]
        npys=[]
        sample = None
        if not self.is_person:
            img=self.load_img(item[0])
            npy=self.load_npy(item[1])
            sample = {'x1': img,'x2':npy,'y':item[-1]}
        else:
            for it in item:
                imgs.append(self.load_img(it[0]))
                npys.append(self.load_npy(it[1]))
                label=it[-1]
            sample = {'x1': imgs,'x2':npys,'y':label}
        return sample

class VoiceDataset(Dataset):
    def __init__(self,train,train_rio,paths,is_person):
        self.items=[]
        for path in paths:
            self.items+=getVoiceSample(path[0],path[1],path[2])
        self.train_rio=int(len(self.items)*train_rio)
        if train:
            self.items=self.items[:self.train_rio]
            self.transform = Compose([Resize([52,52]),RandomCrop([48,48]),RandomHorizontalFlip(),RandomVerticalFlip(),ColorJitter(0.1,0.1,0.1,0.1),ToTensor()])
        else:
            self.items=self.items[self.train_rio:]
            self.transform = Compose([Resize([52,52]),ToTensor()])

        
        if not is_person:
            items=[]
            for person in self.items:
                for img in person:
                    items.append(img)
            self.items=items
        
        self.is_person=is_person
        
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
        imgs=[]
        npys=[]
        label=0.0
        sample = None
        if not self.is_person:
            img=self.load_img(item[0])
            npy=self.load_npy(item[1])
            sample = {'x1': img,'x2':npy,'y':item[-1]}
        else:
            for it in item:
                imgs.append(self.load_img(it[0]))
                npys.append(self.load_npy(it[1]))
                label=it[-1]
            sample = {'x1': imgs,'x2':npys,'y':label}
        return sample
# dataset=BioDataset(1,0.9,"gsr")
# train_dataloader = DataLoader(dataset, batch_size=2,shuffle=True)
# for i,data in enumerate(train_dataloader):
#     x,y=data.values()
#     print(x,y)