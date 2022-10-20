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
from sklearn.cluster import DBSCAN

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
    def __init__(self,extractor,time_extractor,regressor,modal,prototype=None):
        self.extractor=extractor.cuda()
        self.time_extractor=time_extractor.cuda()
        self.regressor=regressor.cuda()
        self.prototype=None
        if prototype:
            self.prototype=prototype.cuda()
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

    def train_init(self,dataset,LR,WEIGHT_DELAY,nets,train_criterion,test_criterion):
        self.dataset=dataset
        self.optimizer = optim.Adam(chain(*[net.parameters() for net in nets]), lr=LR,weight_decay=WEIGHT_DELAY)
        self.testloss_best=1e5
        self.train_criterion=train_criterion
        self.test_criterion=test_criterion
        self.test_criterion.requires_grad_ = False
        self.test_hunxiao=[[],[],[],[],[],[],[],[],[],[]]
        self.train_nets=nets
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
        outputs,fea,_=self.extractor(x)
        return outputs,y

    def prototype_forward(self,outputs,y):
        out,_,fc_w2 = self.prototype(outputs)
        w=fc_w2[0]
        r=2
        prototype_loss_batch = 0
        for type_num in range(self.prototype.outputNum):
            eu_distance = -1*torch.norm(out[0,::] - w[type_num,::].reshape(self.prototype.hiddenNum)) #负的欧式距离
            eu_distance = eu_distance / r
            gussian_distance = torch.exp(eu_distance)
            # if type_num == 0:
            #     max_gussian = gussian_distance
            #     max_id = 0
            # if max_gussian < gussian_distance:
            #     max_gussian = gussian_distance
            #     max_id = type_num
            # yy=torch.log2((1.0+y[0]))
            # prototype_loss_batch=-yy*torch.log(gussian_distance)-(1-yy)*(torch.log(1-gussian_distance))
            if int(float(y[0])//0.1) == type_num:
                prototype_loss = -torch.log(gussian_distance.reshape(1))/self.prototype.outputNum
            else:
                prototype_loss = -torch.log(1-gussian_distance.reshape(1))/self.prototype.outputNum
            prototype_loss_batch += prototype_loss
        return prototype_loss_batch

    def time_extractor_forward(self,data,is_train,is_selfatt,is_dbscan=False):
        prototype_loss=None
        xs,y=data.values()
        x=xs[self.modal]                
        y = y.to(torch.float32).unsqueeze(1)
        if is_train:
            y = y + 0.1*torch.randn(y.size()[0],1)
        y[y<0]=0
        y = y.cuda()
        features = []
        for i,imgs in enumerate(x):
            imgs=imgs.cuda()
            with torch.no_grad():
                _,fea,_=self.extractor(imgs)
            features.append(fea)
        features=torch.stack(features)
        outputs,lstm_output=self.time_extractor(features)
        if self.prototype:
            prototype_loss=self.prototype_forward(outputs,y)
        if not is_dbscan:
            if is_selfatt:
                outputs=self.regressor(outputs)
            else:
                outputs=self.regressor(lstm_output)
        else:
            y=xs[-1]
        return outputs,y,prototype_loss

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

    def time_extractor_train(self,EPOCH,savepath,is_selfatt):
        self.extractor.eval()
        for epoch in range(EPOCH):
            bar = tqdm(total=len(self.dataset.train_dataloader), desc=f"train epoch {epoch}")
            for net in self.train_nets:
                net.train()
            sum_loss=0.0
            sum_pro=0.0
            l1_loss = 0.0
            cnt=0
            for data in self.dataset.train_dataloader:
                if epoch==0:
                    break
                if (data['xs'][0][0]).size()[0]<10:
                    continue
                self.optimizer.zero_grad()
                outputs,y,prototype_loss=self.time_extractor_forward(data,is_train=True,is_selfatt=is_selfatt)
                loss = self.train_criterion(outputs, y)
                weight=0.5
                if prototype_loss is not None:
                    union_loss = weight*loss + (1-weight)*prototype_loss
                    sum_pro+=prototype_loss.item()
                else:
                    union_loss = loss
                union_loss.backward()
                self.optimizer.step()
                sum_loss += loss.item()
                l1_loss += self.test_criterion(outputs, y).item()
                cnt+=1
                bar.set_postfix(**{'MSELoss': sum_loss / cnt, 'PROLoss': sum_pro / cnt, 'mae': l1_loss / cnt})
                bar.update(1)
            bar.close()

            for net in self.train_nets:
                net.eval()
            cnt=0
            l1_loss = 0.0
            sum_pro=0.0
            with torch.no_grad():
                for data in self.dataset.test_dataloader:
                    if (data['xs'][0][0]).size()[0]<10:
                        continue
                    outputs,y,prototype_loss=self.time_extractor_forward(data,is_train=False,is_selfatt=is_selfatt)
                    if prototype_loss is not None:
                        sum_pro+=prototype_loss.item()
                    l1_loss_sub = self.test_criterion(outputs, y).item()
                    l1_loss +=l1_loss_sub
                    cnt+=1
                    self.test_hunxiao[int(y//0.1)].append(l1_loss_sub)
            testloss=l1_loss / cnt
            tmp=[]
            for hunxiao in self.test_hunxiao:
                if len(hunxiao):
                    tmp.append(sum(hunxiao)/len(hunxiao))
            self.test_hunxiao=tmp
            print(self.test_hunxiao)
            
            if testloss < self.testloss_best:
                self.testloss_best = testloss
                state = {
                'extractor': self.extractor.state_dict(),
                'time_extractor': self.time_extractor.state_dict(),
                'regressor': self.regressor.state_dict(),
                'acc': self.testloss_best,
            }
                if self.prototype:
                    state['prototype']=self.prototype.state_dict()
                torch.save(state, savepath)
                
            self.test_hunxiao=[[],[],[],[],[],[],[],[],[],[]]
            print('  [Test] mae: %.03f sum_pro: %.03f [Best] mae: %.03f'% (testloss,sum_pro/cnt,self.testloss_best))

    def feature_space(self,is_selfatt):
        bar = tqdm(total=len(self.dataset.train_dataloader), desc=f"train")
        for net in self.train_nets:
            net.eval()
        space_fea=[]
        space_path=[]
        if os.path.exists('/hdd/sdd/lzq/DLMM_new/dataset/space_fea.npy'):
            space_fea=np.load('/hdd/sdd/lzq/DLMM_new/dataset/space_fea.npy')
            space_path=np.load('/hdd/sdd/lzq/DLMM_new/dataset/space_path.npy')
            space_path=space_path.tolist()
        else:
            with torch.no_grad():
                for data in self.dataset.train_dataloader:
                    
                    if (data['xs'][0][0]).size()[0]<10:
                        continue
                    outputs,path,_=self.time_extractor_forward(data,is_train=False,is_selfatt=is_selfatt,is_dbscan=True)
                    space_fea.append(outputs.cpu().squeeze(0).numpy())
                    space_path.append(path)
                    bar.update(1)
                bar.close()
            space_fea=np.array(space_fea)
            print(space_fea,space_path)
            np.save('/hdd/sdd/lzq/DLMM_new/dataset/space_fea.npy',space_fea)
            np.save('/hdd/sdd/lzq/DLMM_new/dataset/space_path.npy',np.array(space_path))
        self.DBSCAN_space(space_fea,space_path)

    def DBSCAN_space(self,space_fea,space_path):
        clustering = DBSCAN(eps=0.4, min_samples=4).fit(space_fea)
        print(clustering.labels_)
        group={}
        for i,num in enumerate(clustering.labels_):
            if str(num) not in group:
                group[str(num)]=[]
            group[str(num)].append(space_path[i])
        for k,v in group.items():
            print(k,len(v)) 

class TwoModel():
    def __init__(self,FaceModel:SingleModel,VoiceModel:SingleModel,CrossModel:Voice_Time_CrossAttention,regressor=None,biomodel=None):
        self.FaceModel=FaceModel
        self.VoiceModel=VoiceModel
        self.CrossModel=CrossModel.cuda()
        if regressor:
            self.regressor=regressor.cuda()
        if biomodel:
            self.biomodel=biomodel
        

    def load_checkpoint(self,face_checkpoint,voice_checkpoint,cross_checkpoint=None,bio_checkout=None):
        self.VoiceModel.extractor.load_state_dict(voice_checkpoint['net'])

        if cross_checkpoint:
            self.FaceModel.load_time_checkpoint(face_checkpoint)
            self.CrossModel.load_state_dict(cross_checkpoint['cross'])
            self.biomodel.time_extractor.load_state_dict(bio_checkout['time_extractor'])
        else:
            self.FaceModel.extractor.load_state_dict(face_checkpoint['net'])
            

    def train_init(self,dataset,LR,WEIGHT_DELAY,nets):
        self.dataset=dataset
        self.optimizer = optim.Adam(chain(*[net.parameters() for net in nets]), lr=LR,weight_decay=WEIGHT_DELAY)
        self.testloss_best=1e5
        self.train_criterion=nn.MSELoss()
        self.test_criterion=nn.L1Loss()
        self.test_criterion.requires_grad_ = False
        self.test_hunxiao=[[],[],[],[],[],[],[],[],[],[]]

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
            for i in range(1):
                for imgs in xs[i]:
                    imgs=imgs.cuda()
                    _,fea,_=extractor_models[i](imgs)
                    features[i].append(fea)
                features[i]=torch.stack(features[i])
        
            #voice_outputs,_=self.CrossModel(input=features[1],query=features[0])
            face_outputs,_=self.FaceModel.time_extractor(features[0])
            #outputs=torch.cat((face_outputs,voice_outputs),dim=-1)

            if self.biomodel:
                bio_input=xs[2][0].transpose(0,1).unsqueeze(0).cuda()
                bio_ouputs,_=self.biomodel.time_extractor(bio_input)

                outputs=torch.cat((face_outputs,bio_ouputs),dim=-1)
                #outputs=torch.cat((torch.randn((1,128)).cuda(),torch.randn((1,128)).cuda(),torch.randn((1,128)).cuda()),dim=-1)

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
            for imgs in xs[0]:
                imgs=imgs.cuda()
                _,fea,_=extractor_models[0](imgs)
                features[0].append(fea)
            features[0]=torch.stack(features[0])

            for imgs in xs[1]:
                imgs=imgs.cuda()
                _,fea,_=extractor_models[1](imgs)
                features[1].append(fea)
            features[1]=torch.stack(features[1])
        
        outputs,energy=self.CrossModel(input=features[0],query=features[1])
        outputs=self.FaceModel.regressor(outputs)

        return outputs,y

    def train(self,EPOCH,savepath):
        self.FaceModel.extractor.eval()
        self.VoiceModel.extractor.eval()
        self.FaceModel.time_extractor.eval()
        self.CrossModel.eval()
        if self.biomodel:
            self.biomodel.time_extractor.eval()

        for epoch in range(EPOCH):
            bar = tqdm(total=len(self.dataset.train_dataloader), desc=f"train epoch {epoch}")
            self.regressor.train()
            sum_loss=0.0
            l1_loss = 0.0
            cnt=0
            for data in self.dataset.train_dataloader:
                if epoch==0:
                    break
                if (data['xs'][0][0]).size()[0]<10:
                    continue
                self.optimizer.zero_grad()
                outputs,y=self.train_forward(data,is_train=True)
                loss = self.train_criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                sum_loss += loss.item()
                l1_loss += self.test_criterion(outputs, y).item()
                cnt+=1
                bar.set_postfix(**{'Loss': sum_loss / cnt, 'mae': l1_loss / cnt})
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
                    l1_loss_sub = self.test_criterion(outputs, y).item()
                    l1_loss +=l1_loss_sub
                    cnt+=1
                    self.test_hunxiao[int(y//0.1)].append(l1_loss_sub)
            testloss=l1_loss / cnt
            tmp=[]
            for hunxiao in self.test_hunxiao:
                if len(hunxiao):
                    tmp.append(sum(hunxiao)/len(hunxiao))
            self.test_hunxiao=tmp
            print(self.test_hunxiao)
            
            if testloss < self.testloss_best:
                self.testloss_best = testloss
                state = {
                'regressor': self.regressor.state_dict(),
                'acc': self.testloss_best,
                'test_hunxiao':self.test_hunxiao
            }
                torch.save(state, savepath)

            self.test_hunxiao=[[],[],[],[],[],[],[],[],[],[]]
            print('  [Test] mae: %.03f  [Best] mae: %.03f'% (testloss,self.testloss_best))

    def voice_train(self,EPOCH,savepath):
        self.FaceModel.extractor.eval()
        self.VoiceModel.extractor.eval()

        for epoch in range(EPOCH):
            bar = tqdm(total=len(self.dataset.train_dataloader), desc=f"train epoch {epoch}")
            self.CrossModel.train()
            self.FaceModel.regressor.train()
            sum_loss=0.0
            l1_loss = 0.0
            cnt=0
            for data in self.dataset.train_dataloader:
                if epoch==0:
                    break
                if (data['xs'][0][0]).size()[0]<10:
                    continue
                self.optimizer.zero_grad()
                outputs,y=self.time_extractor_forward(data,is_train=True)
                loss = self.train_criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                sum_loss += loss.item()
                l1_loss += self.test_criterion(outputs, y).item()
                cnt+=1
                bar.set_postfix(**{'Loss': sum_loss / cnt, 'mae': l1_loss / cnt})
                bar.update(1)
            bar.close()

            self.CrossModel.eval()
            self.FaceModel.regressor.eval()
            cnt=0
            l1_loss = 0.0
            with torch.no_grad():
                for data in self.dataset.test_dataloader:
                    if (data['xs'][0][0]).size()[0]<10:
                        continue
                    outputs,y=self.time_extractor_forward(data,is_train=False)
                    l1_loss_sub = self.test_criterion(outputs, y).item()
                    l1_loss +=l1_loss_sub
                    cnt+=1
                    self.test_hunxiao[int(y//0.1)].append(l1_loss_sub)
            testloss=l1_loss / cnt
            tmp=[]
            for hunxiao in self.test_hunxiao:
                if len(hunxiao):
                    tmp.append(sum(hunxiao)/len(hunxiao))
            self.test_hunxiao=tmp
            print(self.test_hunxiao)
            
            if testloss < self.testloss_best:
                self.testloss_best = testloss
                state = {
                'cross': self.CrossModel.state_dict(),
                #'voice_extractor': self.VoiceModel.extractor.state_dict(),
                'voice_regressor': self.VoiceModel.regressor.state_dict(),
                'acc': self.testloss_best,
            }
                torch.save(state, savepath)
            self.test_hunxiao=[[],[],[],[],[],[],[],[],[],[]]
            print('  [Test] mae: %.03f  [Best] mae: %.03f'% (testloss,self.testloss_best))

class BioModel():
    def __init__(self,time_extractor,regressor,bio_modal,prototype=None):
        self.time_extractor=time_extractor.cuda()
        self.regressor=regressor.cuda()
        self.prototype=None
        if prototype:
            self.prototype=prototype.cuda()
        self.modal=0 if bio_modal=='ecg' else 1 if bio_modal=='hr' else 2 if bio_modal=='gsr' else -1

    def load_time_checkpoint(self,checkpoint):
        self.time_extractor.load_state_dict(checkpoint['time_extractor'])
        self.regressor.load_state_dict(checkpoint['regressor'])
        self.testloss_best=checkpoint['acc']

    def train_init(self,dataset,LR,WEIGHT_DELAY,nets,train_criterion,test_criterion):
        self.dataset=dataset
        self.optimizer = optim.Adam(chain(*[net.parameters() for net in nets]), lr=LR,weight_decay=WEIGHT_DELAY)
        self.testloss_best=1e5
        self.train_criterion=train_criterion
        self.test_criterion=test_criterion
        self.test_criterion.requires_grad_ = False
        self.test_hunxiao=[[],[],[],[],[],[],[],[],[],[]]
        self.train_nets=nets
        #self.optimizer.add_param_group({'params': self.train_criterion.pain_center, 'lr': 0.001, 'name': 'pain_center'})

    def prototype_forward(self,outputs,y):
        out,_,fc_w2 = self.prototype(outputs)
        w=fc_w2[0]
        r=2
        prototype_loss_batch = 0
        for type_num in range(self.prototype.outputNum):
            eu_distance = -1*torch.norm(out[0,::] - w[type_num,::].reshape(self.prototype.hiddenNum)) #负的欧式距离
            eu_distance = eu_distance / r
            gussian_distance = torch.exp(eu_distance)
            # if type_num == 0:
            #     max_gussian = gussian_distance
            #     max_id = 0
            # if max_gussian < gussian_distance:
            #     max_gussian = gussian_distance
            #     max_id = type_num
            # yy=torch.log2((1.0+y[0]))
            # prototype_loss_batch=-yy*torch.log(gussian_distance)-(1-yy)*(torch.log(1-gussian_distance))
            if int(float(y[0])//0.1) == type_num:
                prototype_loss = -torch.log(gussian_distance.reshape(1))/self.prototype.outputNum
            else:
                prototype_loss = -torch.log(1-gussian_distance.reshape(1))/self.prototype.outputNum
            prototype_loss_batch += prototype_loss
        return prototype_loss_batch

    def time_extractor_forward(self,data,is_train,is_selfatt):
        xs,y=data.values()
        x=xs[2][0][self.modal].unsqueeze(0).unsqueeze(-1) if self.modal>-1 else xs[2][0].transpose(0,1).unsqueeze(0)
        y = y.to(torch.float32).unsqueeze(1)
        if is_train:
            y = y + 0.1*torch.randn(y.size()[0],1)
        y[y<0]=0
        x = x.cuda()
        y = y.cuda()
        outputs,lstm_output=self.time_extractor(x)
        prototype_loss=None
        if self.prototype:
            prototype_loss=self.prototype_forward(outputs,y)
        if is_selfatt:
            outputs=self.regressor(outputs)
        else:
            outputs=self.regressor(lstm_output)
        #print(min_output,outputs,outputs+min_output)
        return outputs,y,prototype_loss

    def time_extractor_train(self,EPOCH,savepath,is_selfatt):
        for epoch in range(EPOCH):
            bar = tqdm(total=len(self.dataset.train_dataloader), desc=f"train epoch {epoch}")
            for net in self.train_nets:
                net.train()
            sum_loss=0.0
            l1_loss = 0.0
            sum_pro=0.0
            cnt=0
            for data in self.dataset.train_dataloader:
                if epoch==0:
                    break
                if data['xs'][2][0].size()[1]<10:
                    continue
                self.optimizer.zero_grad()
                outputs,y,prototype_loss=self.time_extractor_forward(data,is_train=True,is_selfatt=is_selfatt)
                loss = self.train_criterion(outputs, y)
                weight=0.5
                if prototype_loss is not None:
                    union_loss = weight*loss + (1-weight)*prototype_loss
                    sum_pro+=prototype_loss.item()
                else:
                    union_loss = loss
                union_loss.backward()
                self.optimizer.step()
                sum_loss += loss.item()
                l1_loss += self.test_criterion(outputs, y).item()
                cnt+=1
                bar.set_postfix(**{'MSELoss': sum_loss / cnt, 'PROLoss': sum_pro / cnt, 'mae': l1_loss / cnt})
                bar.update(1)
            bar.close()

            for net in self.train_nets:
                net.eval()
            cnt=0
            l1_loss = 0.0
            sum_pro=0.0
            with torch.no_grad():
                for data in self.dataset.test_dataloader:
                    if data['xs'][2][0].size()[1]<10:
                        continue
                    outputs,y,prototype_loss=self.time_extractor_forward(data,is_train=False,is_selfatt=is_selfatt)
                    if prototype_loss is not None:
                        sum_pro+=prototype_loss.item()
                    l1_loss_sub = self.test_criterion(outputs, y).item()
                    l1_loss +=l1_loss_sub
                    cnt+=1
                    self.test_hunxiao[int(y//0.1)].append(l1_loss_sub)
            testloss=l1_loss / cnt
            tmp=[]
            for hunxiao in self.test_hunxiao:
                if len(hunxiao):
                    tmp.append(sum(hunxiao)/len(hunxiao))
            self.test_hunxiao=tmp
            print(self.test_hunxiao)
            
            if testloss < self.testloss_best:
                self.testloss_best = testloss
                state = {
                #'extractor': self.extractor.state_dict(),
                'time_extractor': self.time_extractor.state_dict(),
                'regressor': self.regressor.state_dict(),
                'acc': self.testloss_best,
            }
                if self.prototype:
                    state['prototype']=self.prototype.state_dict()
                torch.save(state, savepath)
            self.test_hunxiao=[[],[],[],[],[],[],[],[],[],[]]
            print('  [Test] mae: %.03f sum_pro: %.03f [Best] mae: %.03f'% (testloss,sum_pro/cnt,self.testloss_best))