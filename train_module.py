from tkinter import W
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import argparse
import numpy as np
from itertools import chain
import copy
from tqdm import tqdm
import os
import sys
from torch.utils.data import DataLoader
from utils import getCfg
from models import *
from loader import BioDataset,FaceDataset,VoiceDataset,AllDataset
import pdb
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_sequence

class DataSet():
    def __init__(self,batch_size,TRAIN_RIO,DATA_PATHS,modal,is_time,collate_fn,pic_size):
        train_dataset=AllDataset(1,TRAIN_RIO,DATA_PATHS,modal,is_time,pic_size)
        test_dataset=AllDataset(0,TRAIN_RIO,DATA_PATHS,modal,is_time,pic_size)
        if collate_fn is None:
            self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
            self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)
        else:
            self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,collate_fn=collate_fn)
            self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False,collate_fn=collate_fn)

class SingleModel():
    def __init__(self,extractor,time_extractor,regressor,modal,classifier=None):
        self.extractor=extractor.cuda()
        self.time_extractor=time_extractor.cuda()
        self.regressor=regressor.cuda()
        if classifier:
            self.classifier=classifier.cuda()
        self.modal=0 if modal=='face' else 1 if modal=='voice' else -1 if modal=='face_point' else 2

    def load_checkpoint(self,checkpoint):
        self.extractor.load_state_dict(checkpoint['net'])
        self.testloss_best=checkpoint['acc']
        print(checkpoint['acc'])

    def load_time_checkpoint(self,checkpoint):
        self.extractor.load_state_dict(checkpoint['extractor'])
        self.time_extractor.load_state_dict(checkpoint['time_extractor'])
        self.regressor.load_state_dict(checkpoint['regressor'])
        self.testloss_best=checkpoint['acc']
        print(checkpoint['acc'])

    def train_init(self,dataset,LR,WEIGHT_DELAY,nets,train_criterion,test_criterion):
        self.dataset=dataset
        self.optimizer = optim.Adam(chain(*[net.parameters() for net in nets]), lr=LR,weight_decay=WEIGHT_DELAY)
        self.testloss_best=1e5
        self.train_criterion=train_criterion
        self.test_criterion=test_criterion
        self.test_criterion.requires_grad_ = False
        #self.optimizer.add_param_group({'params': self.train_criterion.pain_center, 'lr': 0.001, 'name': 'pain_center'})

    def classifier_forward(self,data):
        xs,y=data.values()
        x=xs[self.modal]                
        y = y.to(torch.float32)
        y[y<=0.2] = 0
        y[y>0.2] = 1
        y=y.to(torch.long)
        x, y = x.cuda(), y.cuda()
        _,fea,_=self.extractor(x)
        outputs,_=self.classifier(fea)
        return outputs,y

    def extractor_forward(self,data,is_train):
        xs,y=data.values()
        x=xs[self.modal]                
        y = y.to(torch.float32).unsqueeze(1)
        if is_train:
            y = y + 0.1*torch.randn(y.size()[0],1)
        y[y<0] = 0
        x, y = x.cuda(), y.cuda()
        outputs,_,_=self.extractor(x)
        return outputs,y

    def time_extractor_forward(self,data,is_train,is_extractor_train):
        xs,y=data.values()
        x=xs[self.modal]                
        y = y.to(torch.float32).unsqueeze(1)
        if is_train:
            y = y + 0.1*torch.randn(y.size()[0],1)
        y[y<0]=0
        y = y.cuda()
        features = []
        min_num=0
        min_output=0
        for i,imgs in enumerate(x):
            imgs=imgs.cuda()
            if is_extractor_train:
                _,fea=self.extractor(imgs)
                #fea=self.extractor(imgs[:,0,:],imgs[:,1,:])
            else:
                with torch.no_grad():
                    _,fea,_=self.extractor(imgs)
            # outputs=outputs.squeeze(-1)
            # min_num=outputs.argmin()
            # min_output=outputs[min_num]
            features.append(fea)
        # features.sort(key=lambda x: len(x), reverse=True)
        # features = pack_sequence(features).float()
        features=torch.stack(features).squeeze(0)
        # min_fea=features[min_num].clone()
        # features-=min_fea
        features=features.unsqueeze(0)
        outputs,energy=self.time_extractor(features)
        outputs=self.regressor(outputs)
        #print(min_output,outputs,outputs+min_output)
        return outputs,y

    def classifier_test(self):
        self.extractor.eval()
        self.classifier.eval()
        correct = 0.0
        total=0
        with torch.no_grad():
            for _,data in enumerate(self.dataset.test_dataloader):
                outputs,y=self.classifier_forward(data)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        return correct/total

    def extractor_test(self,criterion):
        self.extractor.eval()
        cnt=0
        sum_loss = 0.0
        with torch.no_grad():
            for data in self.dataset.test_dataloader:
                outputs,y=self.extractor_forward(data,is_train=False)
                loss = criterion(outputs, y)
                sum_loss+=loss.item()
                cnt+=1
        return sum_loss/cnt

    def classifier_train(self,EPOCH,savepath):
        for epoch in range(EPOCH):
            bar = tqdm(total=len(self.dataset.train_dataloader), desc=f"train epoch {epoch}")
            self.extractor.train()
            self.classifier.train()
            sum_loss=0.0
            correct = 0.0
            total=0
            for _,data in enumerate(self.dataset.train_dataloader):
                self.optimizer.zero_grad()
                outputs,y=self.classifier_forward(data)
                loss = self.train_criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                sum_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                #l1_loss += self.test_criterion(outputs, y).item()
                bar.set_postfix(**{'Loss': sum_loss / total, 'acc': correct / total})
                bar.update(1)
            bar.close()
            
            testloss=self.classifier_test()
            
            if testloss > self.testloss_best:
                self.testloss_best = testloss
                state = {
                'extractor': self.extractor.state_dict(),
                'classifier': self.classifier.state_dict(),
                'acc': self.testloss_best,
            }
                torch.save(state, savepath)
            print('  [Test] acc: %.03f  [Best] acc: %.03f'% (testloss,self.testloss_best))

    def extractor_train(self,EPOCH,savepath):
        for epoch in range(EPOCH):
            bar = tqdm(total=len(self.dataset.train_dataloader), desc=f"train epoch {epoch}")
            self.extractor.train()
            sum_loss=0.0
            l1_loss = 0.0
            for i,data in enumerate(self.dataset.train_dataloader):
                self.optimizer.zero_grad()
                outputs,y=self.extractor_forward(data,is_train=True)
                loss = self.train_criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                sum_loss += loss.item()
                l1_loss += self.test_criterion(outputs, y).item()
                bar.set_postfix(**{'Loss': sum_loss / (i + 1), 'mae': l1_loss / (i + 1)})
                bar.update(1)
            bar.close()
            
            testloss=self.extractor_test(self.test_criterion)
            
            if testloss < self.testloss_best:
                self.testloss_best = testloss
                state = {
                'net': self.extractor.state_dict(),
                'acc': self.testloss_best,
            }
                torch.save(state, savepath)
            print('  [Test] mae: %.03f  [Best] mae: %.03f'% (testloss,self.testloss_best))

    def time_extractor_train(self,EPOCH,savepath,is_extractor_train):
        for epoch in range(EPOCH):
            bar = tqdm(total=len(self.dataset.train_dataloader), desc=f"train epoch {epoch}")
            if is_extractor_train:
                self.extractor.train()
            else:
                self.extractor.eval()
            self.time_extractor.train()
            self.regressor.train()
            sum_loss=0.0
            l1_loss = 0.0
            for i,data in enumerate(self.dataset.train_dataloader):
                #print((data['xs'][0][0]).size()[0])
                if (data['xs'][0][0]).size()[0]<10:
                    continue
                self.optimizer.zero_grad()
                outputs,y=self.time_extractor_forward(data,is_train=True,is_extractor_train=is_extractor_train)
                loss = self.train_criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                sum_loss += loss.item()
                l1_loss += self.test_criterion(outputs, y).item()
                bar.set_postfix(**{'Loss': sum_loss / (i + 1), 'mae': l1_loss / (i + 1)})
                bar.update(1)
            bar.close()

            self.extractor.eval()
            self.time_extractor.eval()
            self.regressor.eval()
            cnt=0
            l1_loss = 0.0
            with torch.no_grad():
                for data in self.dataset.test_dataloader:
                    if (data['xs'][0][0]).size()[0]<10:
                        continue
                    outputs,y=self.time_extractor_forward(data,is_train=False,is_extractor_train=is_extractor_train)
                    l1_loss += self.test_criterion(outputs, y).item()
                    cnt+=1
            testloss=l1_loss / cnt
            
            if testloss < self.testloss_best:
                self.testloss_best = testloss
                state = {
                'extractor': self.extractor.state_dict(),
                'time_extractor': self.time_extractor.state_dict(),
                'regressor': self.regressor.state_dict(),
                'acc': self.testloss_best,
            }
                torch.save(state, savepath)
            print('  [Test] mae: %.03f  [Best] mae: %.03f'% (testloss,self.testloss_best))


class TwoModel():
    def __init__(self,FaceModel:SingleModel,VoiceModel:SingleModel,CrossModel:Voice_Time_CrossAttention,regressor=None):
        self.FaceModel=FaceModel
        self.VoiceModel=VoiceModel
        self.CrossModel=CrossModel.cuda()
        if regressor:
            self.regressor=regressor.cuda()

    def load_checkpoint(self,face_checkpoint,voice_checkpoint,cross_checkpoint=None):
        self.FaceModel.load_time_checkpoint(face_checkpoint)
        self.VoiceModel.load_time_checkpoint(voice_checkpoint)
        if cross_checkpoint:
            self.CrossModel.load_state_dict(cross_checkpoint['cross'])
            print(cross_checkpoint['acc'])

    def train_init(self,dataset,LR,WEIGHT_DELAY,nets):
        self.dataset=dataset
        self.optimizer = optim.Adam(chain(*[net.parameters() for net in nets]), lr=LR,weight_decay=WEIGHT_DELAY)
        self.testloss_best=1e5
        self.train_criterion=nn.MSELoss()
        self.test_criterion=nn.L1Loss()
        self.test_criterion.requires_grad_ = False

    def train_forward(self,data,is_train):
        xs,y=data.values()                
        y = y.to(torch.float32).unsqueeze(1)
        if is_train:
            y = y + 0.1*torch.randn(y.size()[0],1)
        y[y<0]=0
        y = y.cuda()

        features = [[],[]]
        extractor_models=[self.FaceModel.extractor,self.VoiceModel.extractor]
        with torch.no_grad():
            for i in range(2):
                for imgs in xs[i]:
                    imgs=imgs.cuda()
                    _,fea,_=extractor_models[i](imgs)
                    features[i].append(fea)
                features[i]=torch.stack(features[i])
        
            voice_outputs,_=self.CrossModel(features[1],features[0])
            face_outputs,_=self.FaceModel.time_extractor(features[0])
            outputs=torch.cat((face_outputs,voice_outputs),dim=-1)

        outputs=self.regressor(outputs)

        return outputs,y

    def time_extractor_forward(self,data,is_train):
        xs,y=data.values()                
        y = y.to(torch.float32).unsqueeze(1)
        if is_train:
            y = y + 0.1*torch.randn(y.size()[0],1)
        y[y<0]=0
        y = y.cuda()

        features = [[],[]]
        extractor_models=[self.FaceModel.extractor,self.VoiceModel.extractor]
        with torch.no_grad():
            for i in range(2):
                for imgs in xs[i]:
                    imgs=imgs.cuda()
                    _,fea,_=extractor_models[i](imgs)
                    features[i].append(fea)
                features[i]=torch.stack(features[i])
        
        outputs,energy=self.CrossModel(features[1],features[0])
        outputs=self.VoiceModel.regressor(outputs)

        return outputs,y

    def train(self,EPOCH,savepath):
        self.FaceModel.extractor.eval()
        self.VoiceModel.extractor.eval()
        self.FaceModel.time_extractor.eval()
        self.CrossModel.eval()

        for epoch in range(EPOCH):
            bar = tqdm(total=len(self.dataset.train_dataloader), desc=f"train epoch {epoch}")
            self.regressor.train()
            sum_loss=0.0
            l1_loss = 0.0
            for i,data in enumerate(self.dataset.train_dataloader):
                #print((data['xs'][0][0]).size()[0])
                if (data['xs'][0][0]).size()[0]<10:
                    continue
                self.optimizer.zero_grad()
                outputs,y=self.train_forward(data,is_train=True)
                loss = self.train_criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                sum_loss += loss.item()
                l1_loss += self.test_criterion(outputs, y).item()
                bar.set_postfix(**{'Loss': sum_loss / (i + 1), 'mae': l1_loss / (i + 1)})
                bar.update(1)
            bar.close()

            self.regressor.eval()
            cnt=0
            l1_loss = 0.0
            with torch.no_grad():
                for data in self.dataset.test_dataloader:
                    if (data['xs'][0][0]).size()[0]<10:
                        continue
                    outputs,y=self.train_forward(data,is_train=False)
                    l1_loss += self.test_criterion(outputs, y).item()
                    cnt+=1
            testloss=l1_loss / cnt
            
            if testloss < self.testloss_best:
                self.testloss_best = testloss
                state = {
                'regressor': self.regressor.state_dict(),
                'acc': self.testloss_best,
            }
                torch.save(state, savepath)
            print('  [Test] mae: %.03f  [Best] mae: %.03f'% (testloss,self.testloss_best))

    def voice_train(self,EPOCH,savepath):
        self.FaceModel.extractor.eval()
        self.VoiceModel.extractor.eval()

        for epoch in range(EPOCH):
            bar = tqdm(total=len(self.dataset.train_dataloader), desc=f"train epoch {epoch}")
            self.CrossModel.train()
            self.VoiceModel.regressor.train()
            sum_loss=0.0
            l1_loss = 0.0
            for i,data in enumerate(self.dataset.train_dataloader):
                #print((data['xs'][0][0]).size()[0])
                if (data['xs'][0][0]).size()[0]<10:
                    continue
                self.optimizer.zero_grad()
                outputs,y=self.time_extractor_forward(data,is_train=True)
                loss = self.train_criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                sum_loss += loss.item()
                l1_loss += self.test_criterion(outputs, y).item()
                bar.set_postfix(**{'Loss': sum_loss / (i + 1), 'mae': l1_loss / (i + 1)})
                bar.update(1)
            bar.close()

            self.CrossModel.eval()
            self.VoiceModel.regressor.eval()
            cnt=0
            l1_loss = 0.0
            with torch.no_grad():
                for data in self.dataset.test_dataloader:
                    if (data['xs'][0][0]).size()[0]<10:
                        continue
                    outputs,y=self.time_extractor_forward(data,is_train=False)
                    l1_loss += self.test_criterion(outputs, y).item()
                    cnt+=1
            testloss=l1_loss / cnt
            
            if testloss < self.testloss_best:
                self.testloss_best = testloss
                state = {
                'cross': self.CrossModel.state_dict(),
                'voice_regressor': self.VoiceModel.regressor.state_dict(),
                'acc': self.testloss_best,
            }
                torch.save(state, savepath)
            print('  [Test] mae: %.03f  [Best] mae: %.03f'% (testloss,self.testloss_best))