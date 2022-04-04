import torch
import numpy as np
from PIL import Image
from utils import getSample,readCsv,fileFeatureExtraction,getFaceSample,getBioSample,extractGsr,npyStandard,getVoiceSample
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import ToTensor, Resize, RandomCrop,Compose,RandomHorizontalFlip,RandomVerticalFlip,ColorJitter
import random
import os
from models import Prototype,Classifier,ResNet18,cnn1d,VGG,Regressor
import torchvision.transforms.functional as tf

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
    def __init__(self,train,train_rio,paths,is_person,tcn_num=1,person_test=0):
        self.train=train
        self.is_person=is_person
        self.items=[]
        self.person_test=person_test

        for path in paths:
            self.items+=getFaceSample(path[0],path[1],path[2])

        self.train_rio=int(len(self.items)*train_rio)

        if train:
            self.items=self.items[:self.train_rio]
            self.transform = Compose([Resize([52,52]),RandomCrop([48,48]),RandomHorizontalFlip(),RandomVerticalFlip(),ToTensor()])
        else:
            self.items=self.items[self.train_rio:]
            self.transform = Compose([Resize([48,48]),ToTensor()])

        items=[]
        itemss=[]
        
        if not is_person:
            for person in self.items:
                for img in person:
                    items.append(img)
            self.items=items
        else:
            self.transform = Compose([Resize([48,48]),ToTensor()])
            
            for person in self.items:
                i=0
                LEN=len(person)

                while i<LEN:
                    item=[]
                    choose_num=[x for x in range(i,i+min(tcn_num,LEN-i))]
                    if (LEN-i)<tcn_num:
                        for _ in range(tcn_num-(LEN-i)):
                            randint=random.randint(i,LEN-1)
                            choose_num.append(randint)
                        choose_num.sort()

                    for k in choose_num:
                        item.append(person[k])
                    items.append(item)

                    i+=tcn_num

                if person_test:
                    itemss.append(items)
                    items=[]

        self.items=items
        if person_test:
            self.items=itemss
        print(len(self.items))

    def load_img(self,file_path,hf=0.0,vf=0.0):
        img = Image.open(file_path).convert('L')
        img=np.array(img)
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        img=self.transform(img)
        if hf>0.5:
            img=tf.hflip(img)
        if vf>0.5:
            img=tf.vflip(img)
        return img

    def load_npy(self,file_path):
        npy=np.load(file_path)
        npy=npyStandard(npy)
        npy=torch.from_numpy(npy)
        npy=npy.to(torch.float32)
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
            if not self.person_test:
                hf,vf=0.0,0.0
                if self.train:
                    hf=random.random()
                    vf=random.random()
                for it in item:
                    imgs.append(self.load_img(it[0],hf,vf))
                    npys.append(self.load_npy(it[1]))
                    label=it[-1]
                imgs=torch.stack(imgs)
                npys=torch.stack(npys)
                sample = {'x1': imgs,'x2':npys,'y':label}
            else:
                imgss=[]
                npyss=[]
                for ite in item:
                    imgs=[]
                    npys=[]
                    for it in ite:
                        imgs.append(self.load_img(it[0]))
                        npys.append(self.load_npy(it[1]))
                        label=it[-1]
                    imgs=torch.stack(imgs)
                    npys=torch.stack(npys)
                    imgss.append(imgs)
                    npyss.append(npys)
                sample = {'x1': imgss,'x2':npyss,'y':label}
        return sample

class VoiceDataset(Dataset):
    def __init__(self,train,train_rio,paths,is_person,tcn_num=1,person_test=0):
        self.train=train
        self.is_person=is_person
        self.items=[]
        self.person_test=person_test

        for path in paths:
            self.items+=getVoiceSample(path[0],path[1],path[2])

        self.train_rio=int(len(self.items)*train_rio)

        if train:
            self.items=self.items[:self.train_rio]
            
        else:
            self.items=self.items[self.train_rio:]
            

        items=[]
        itemss=[]
        
        if not is_person:
            for person in self.items:
                for wav in person:
                    items.append(wav)
            self.items=items
        else:
            for person in self.items:
                i=0
                LEN=len(person)

                while i<LEN:
                    item=[]
                    choose_num=[x for x in range(i,i+min(tcn_num,LEN-i))]
                    if (LEN-i)<tcn_num:
                        for _ in range(tcn_num-(LEN-i)):
                            randint=random.randint(i,LEN-1)
                            choose_num.append(randint)
                        choose_num.sort()

                    for k in choose_num:
                        item.append(person[k])
                    items.append(item)

                    i+=tcn_num

                if person_test:
                    itemss.append(items)
                    items=[]

        self.items=items
        if person_test:
            self.items=itemss
        print(len(self.items))

    def load_npy(self,file_path):
        npy=np.load(file_path)
        npy=torch.from_numpy(npy)
        npy=torch.unsqueeze(npy, 0)
        npy=npy.to(torch.float32)
        return npy

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idr):
        item=self.items[idr]
        npys=[]
        label=0.0
        sample = None
        if not self.is_person:
            npy=self.load_npy(item[0])
            sample = {'x':npy,'y':item[-1]}
        else:
            for it in item:
                npys.append(self.load_npy(it[0]))
                label=it[-1]
            
            npys=torch.stack(npys)
            sample = {'x':npys,'y':label}
        return sample