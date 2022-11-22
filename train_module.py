import csv
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
from sklearn.cluster import DBSCAN,KMeans
from sklearn.manifold import TSNE
from typing import List
import time
from scipy.signal import savgol_filter

class DataSet():
    def __init__(self,batch_size,TRAIN_RIO,DATA_PATHS,modal,is_time,collate_fn,pic_size):
        train_dataset=AllDataset(1,TRAIN_RIO,DATA_PATHS,modal,is_time,pic_size)
        test_dataset=AllDataset(0,TRAIN_RIO,DATA_PATHS,modal,is_time,pic_size)
        train_dataset.get_y_label()
        test_dataset.get_y_label()
        if collate_fn is None:
            self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
            self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)
        else:
            self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,collate_fn=collate_fn)
            self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False,collate_fn=collate_fn)

class SingleModel():
    def __init__(self,extractor,time_extractor,regressor,modal,prototype=None,cluster=None):
        self.extractor=extractor.cuda()
        self.time_extractor=time_extractor.cuda()
        self.regressor=regressor.cuda()
        self.prototype=None
        if prototype:
            self.prototype=prototype.cuda()
        self.cluster=None
        if cluster:
            self.cluster=cluster.cuda()
        self.forward_nets=[self.extractor,self.time_extractor,self.regressor]
        self.modal=0 if modal=='face' else 1 if modal=='voice' else -1 if modal=='face_point' else 2

    def load_checkpoint(self,checkpoint):
        self.extractor.load_state_dict(checkpoint['net'])
        self.testloss_best=checkpoint['acc']
        print(checkpoint['acc'])

    def load_time_checkpoint(self,checkpoint):
        self.extractor.load_state_dict(checkpoint['extractor'])
        self.time_extractor.load_state_dict(checkpoint['time_extractor'])
        self.regressor.load_state_dict(checkpoint['regressor'])
        #print(checkpoint['acc'])
        #self.testloss_best=checkpoint['acc']

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
                outputs=F.relu(self.regressor(outputs))
            else:
                outputs=F.relu(self.regressor(lstm_output))
        else:
            prototype_loss=xs[-1]
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
            bar = tqdm(total=len(self.dataset.test_dataloader), desc=f"test epoch {epoch}")
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
                    bar.set_postfix(**{'mae': l1_loss / cnt})
                    bar.update(1)
            bar.close()
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

    def feature_space(self,is_selfatt,savepath):
        bar = tqdm(total=len(self.dataset.train_dataloader), desc=f"feature_space")
        for net in self.forward_nets:
            net.eval()
        space_fea=[]
        space_y=[]
        space_path=[]
        if os.path.exists(os.path.join(savepath,'space_fea.pt')):
            space_fea=torch.load(os.path.join(savepath,'space_fea.pt'))
            space_y=torch.load(os.path.join(savepath,'space_y.pt'))
            space_path=np.load(os.path.join(savepath,'space_path.npy'))
            space_path=space_path.tolist()
        else:
            with torch.no_grad():
                for data in self.dataset.train_dataloader:
                    if (data['xs'][0][0]).size()[0]<10:
                        continue
                    outputs,y,path=self.time_extractor_forward(data,is_train=False,is_selfatt=is_selfatt,is_dbscan=True)
                    space_fea.append(outputs.cpu().squeeze(0))
                    space_y.append(y.cpu().squeeze(0))
                    space_path.append(path)
                    bar.update(1)
                bar.close()
            space_fea=torch.stack(space_fea)
            space_y=torch.stack(space_y)
            torch.save(torch.stack(space_fea),os.path.join(savepath,'space_fea.pt'))
            torch.save(torch.stack(space_y),os.path.join(savepath,'space_y.pt'))
            np.save(os.path.join(savepath,'space_path.npy'),np.array(space_path))
        return self.cluster_space(space_fea.cuda(),space_y.cuda(),space_path,savepath)
        

    def cluster_space(self,space_fea,space_y,space_path,savepath):
        if os.path.exists(os.path.join(savepath,'cluster_centerList.npy')):
            space_path=np.load(os.path.join(savepath,'cluster_space_path.npy'), allow_pickle='TRUE')
            space_path=space_path.item()
            centerList=np.load(os.path.join(savepath,'cluster_centerList.npy'))

        else:
            for net in self.train_nets:
                net.train()
            for net in self.forward_nets:
                net.eval()
            
            EPOCH_SIZE=1000
            bar = tqdm(total=EPOCH_SIZE, desc=f"cluster_space")
            for epoch in range(EPOCH_SIZE):
                self.optimizer.zero_grad()
                loss_fea=[]
                loss_y=[]
                cluster_labels=[0]*space_fea.shape[0]
                cluster_y=[0,0]
                group={}

                _,center=self.cluster(space_fea[0])
                with torch.no_grad():
                    for k in range(2):
                        cluster_y[k]=self.regressor(center[0][k].unsqueeze(0))

                for i in range(space_fea.shape[0]):
                    fea,center=self.cluster(space_fea[i])
                    gussian_distance=[]
                    for k in range(2):
                        eu_distance = -1*torch.norm(fea - center[0][k]) #负的欧式距离
                        gussian_distance.append(torch.exp(eu_distance))
                    maxn_k=0 if gussian_distance[0]>gussian_distance[1] else 1
                    subloss=-torch.log(gussian_distance[maxn_k])-torch.log(1-gussian_distance[1-maxn_k])
                    loss_fea.append(subloss)
                    loss_y.append((space_y[i]-cluster_y[maxn_k])**2)
                    
                    cluster_labels[i]=maxn_k
                    group[space_path[i][0]]=maxn_k
                
                loss_center=[]
                for k in range(2):
                    _,center=self.cluster(space_fea[0])
                    eu_distance = -1*torch.norm(center[0][k]-torch.mean(space_fea))
                    gussian_distance=torch.exp(eu_distance)
                    loss_center.append(-torch.log(1-gussian_distance))
                loss_center=torch.mean(torch.stack(loss_center))
                loss_y=torch.mean(torch.stack(loss_y))
                loss_fea=torch.mean(torch.stack(loss_fea))
                # for k in range(2):
                #     y_std=torch.std(torch.stack(cluster_y[k]))
                #     loss+=y_std
                loss_weight=0.1
                loss=loss_fea+loss_y+loss_center
                loss.backward()
                self.optimizer.step()
                bar.set_postfix(**{'Loss': loss.cpu().item()})
                bar.update(1)
            bar.close()
            
            _,centerList=self.cluster(space_fea[0])
            space_path=group

            space_fea=space_fea.cpu().detach().numpy()
            space_y=space_y.cpu().detach().numpy()
            cluster_labels=np.array(cluster_labels)
            centerList=centerList[0].cpu().detach().numpy()
            self.tsne_space(space_fea,space_y,cluster_labels,centerList,savepath)
                            
                
            np.save(os.path.join(savepath,'cluster_space_path.npy'),space_path)
            np.save(os.path.join(savepath,'cluster_centerList.npy'),centerList)
        
        return space_path,centerList

    def tsne_space(self,space_fea,space_y,labels,centerList,savepath):
        #tsne_fea=TSNE().fit_transform(np.concatenate([space_fea,centerList],axis=0))
        if os.path.exists(os.path.join(savepath,'tsne_fea.npy')):
            tsne_fea=np.load(os.path.join(savepath,'tsne_fea.npy'))
        else:
            tsne_fea=TSNE().fit_transform(space_fea)
            np.save(os.path.join(savepath,'tsne_fea.npy'),tsne_fea)
        tsne_fea_list=[[],[],[]]
        color=['red','green','blue']
        for i in range(len(labels)):
            tsne_fea_list[labels[i]].append(tsne_fea[i])
        for i in range(2):
            dots=np.array(tsne_fea_list[i])
            plt.scatter(dots[:,0],dots[:,1],c=color[i])
            #plt.scatter([tsne_fea[-(2-i)][0]],[tsne_fea[-(2-i)][1]],c=color[i],s=1000,alpha=0.3)
        plt.savefig(os.path.join(savepath,'tsne_cluster.jpg'))
        plt.close()

        # tsne_y_list=[[],[],[],[],[],[],[],[],[]]
        # for i in range(len(labels)):
        #     tsne_y_list[int(space_y[i]//0.1)].append(tsne_fea[i])
        # for i in range(9):
        #     dots_y=np.array(tsne_y_list[i])
        plt.scatter(tsne_fea[:,0],tsne_fea[:,1],c=space_y//0.1*0.1,cmap="coolwarm")
        plt.savefig(os.path.join(savepath,'tsne_y.jpg'))
        plt.close()


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
        self.FaceModel.load_time_checkpoint(face_checkpoint)
        self.VoiceModel.load_time_checkpoint(voice_checkpoint)
        self.biomodel.load_time_checkpoint(bio_checkout)
        # self.VoiceModel.extractor.load_state_dict(voice_checkpoint['net'])

        # if cross_checkpoint:
        #     self.FaceModel.load_time_checkpoint(face_checkpoint)
        #     self.CrossModel.load_state_dict(cross_checkpoint['cross'])
        #     self.biomodel.time_extractor.load_state_dict(bio_checkout['time_extractor'])
        # else:
        #     self.FaceModel.extractor.load_state_dict(face_checkpoint['net'])
            

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
        outputs_list=[]
        models=[self.FaceModel,self.VoiceModel]
        with torch.no_grad():
            for i in range(2):
                for imgs in xs[i]:
                    imgs=imgs.cuda()
                    _,fea,_=models[i].extractor(imgs)
                    features[i].append(fea)
                features[i]=torch.stack(features[i])
                time_outputs,_=models[i].time_extractor(features[i])
                outputs_list.append(models[i].regressor(time_outputs))
        
            #voice_outputs,_=self.CrossModel(input=features[1],query=features[0])
            
            outputs=torch.cat((outputs_list[0],outputs_list[1]),dim=-1)

            if self.biomodel:
                bio_input=xs[2][0].transpose(0,1).unsqueeze(0).cuda()
                bio_ouputs,_=self.biomodel.time_extractor(bio_input)
                outputs_list.append(self.biomodel.regressor(bio_ouputs))

                outputs=torch.cat((outputs_list[0],outputs_list[1],outputs_list[2]),dim=-1)

                #outputs=torch.cat((torch.randn((1,128)).cuda(),torch.randn((1,128)).cuda(),torch.randn((1,128)).cuda()),dim=-1)

        outputs,att=self.regressor(outputs)

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
        self.FaceModel.time_extractor.eval()
        self.FaceModel.regressor.eval()

        self.VoiceModel.extractor.eval()
        self.VoiceModel.time_extractor.eval()
        self.VoiceModel.regressor.eval()

        self.CrossModel.eval()

        if self.biomodel:
            self.biomodel.time_extractor.eval()
            self.biomodel.regressor.eval()

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
        print(checkpoint['acc'])
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

class MultiExperts():
    def __init__(self,modelList:List[SingleModel],backbone:SingleModel):
        self.modelList=modelList
        for i in range(len(self.modelList)):
            self.modelList[i].extractor.cuda()
            self.modelList[i].time_extractor.cuda()
            self.modelList[i].regressor.cuda()

        self.backbone=backbone
        self.backbone.extractor.cuda()
        self.backbone.time_extractor.cuda()
        self.backbone.regressor.cuda()

    def load_checkpoint(self,checkpointList,backboneCheckpoint,space_path=None,centerList=None):
        for i in range(len(self.modelList)):
            self.modelList[i].load_time_checkpoint(checkpointList[i])
        self.backbone.load_time_checkpoint(backboneCheckpoint)
        self.centerList=[]
        self.stdList=[]
        self.space_path=[0,0,1,1,1,1,2,2,2]
        if space_path:
            self.space_path=space_path
        if centerList:
            self.centerList=torch.from_numpy(centerList).cuda()
        
    def train_init(self,dataset,LR,WEIGHT_DELAY):
        self.dataset=dataset
        self.optimizerList = []
        nets=[self.modelList[0].time_extractor]
        for i in range(len(self.modelList)):
            nets+=[self.modelList[i].regressor]
        opt=optim.Adam(chain(*[net.parameters() for net in nets]), lr=LR,weight_decay=WEIGHT_DELAY)
        self.optimizer=opt
        
        self.train_criterion=nn.MSELoss()
        self.test_criterion=nn.L1Loss()
        self.test_criterion.requires_grad_ = False
        self.testloss_best=1e5
        self.test_hunxiao=[[],[],[],[],[],[],[],[],[],[]]

    def train_forward(self,data,is_train,model:SingleModel):
        backbone=self.modelList[0]
        xs,y=data.values()                
        y = y.to(torch.float32).unsqueeze(1)
        if is_train:
            y = y + 0.1*torch.randn(y.size()[0],1)
        y[y<0]=0
        y = y.cuda()

        features = []
        with torch.no_grad():
            for imgs in xs[model.modal]:
                imgs=imgs.cuda()
                _,fea,_=backbone.extractor(imgs)
                features.append(fea)
            features=torch.stack(features)
        time_outputs,_=backbone.time_extractor(features)
        outputs=F.relu(model.regressor(time_outputs))

        return outputs,y,time_outputs

    def backbone_forward(self,xs):
        self.backbone.extractor.eval()
        self.backbone.time_extractor.eval()
        features = []
        with torch.no_grad():
            for imgs in xs:
                imgs=imgs.cuda()
                _,fea,_=self.backbone.extractor(imgs)
                features.append(fea)
            features=torch.stack(features)
            time_outputs,_=self.backbone.time_extractor(features)
        return time_outputs

    def GuassianDist(self,x,y,s):
        eu_distance = -1*torch.norm((x-y)/s) #负的欧式距离
        gussian_distance = torch.exp(eu_distance)
        return gussian_distance

    def whichCluster(self,x):
        dis_list=[]
        #x_fea=self.backbone_forward(x)
        for i in range(len(self.modelList)):
            dis=self.GuassianDist(x[i],self.centerList[i],self.stdList[i])
            dis_list.append(dis)
        weights=[]
        for i in range(len(self.modelList)):
            weights.append(dis_list[i]/sum(dis_list))
        #maxn=1
        # max_id=weights.index(max(weights))
        # for i in range(len(self.modelList)):
        #     weights[i]=1 if i==max_id else 0

        return weights
    
    def loss_visualize(self,epoch, plt_loss_list):
        epochs=[i for i in range(epoch+1)]
        plt.style.use("ggplot")
        plt.figure()
        plt.title("Epoch_Loss")
        plt.plot(epochs, plt_loss_list[0], label='red_loss', color='r', linestyle='-')
        plt.plot(epochs, plt_loss_list[1], label='green_loss', color='g', linestyle='-')
        #plt.plot(epochs, plt_loss_list[2], label='blue_loss', color='b', linestyle='-')
        #plt.plot(epochs, plt_loss_list[3], label='total_loss', color='purple', linestyle='-')
        plt.plot(epochs, plt_loss_list[-1], label='val_loss', color='y', linestyle='-')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig('/hdd/sdd/lzq/DLMM_new/model/loss.jpg')
        plt.close()

    def test_init(self,checkpoint,dataset):
        print(checkpoint["acc"],checkpoint["test_hunxiao"])
        for i in range(len(self.modelList)):
            self.modelList[i].load_time_checkpoint(checkpoint["modelList"][i])
            self.modelList[i].extractor.eval()
            self.modelList[i].time_extractor.eval()
            self.modelList[i].regressor.eval()

        self.backbone.load_time_checkpoint(checkpoint["backbone"])
        self.backbone.extractor.eval()
        self.backbone.time_extractor.eval()
        self.backbone.regressor.eval()

        self.centerList=checkpoint["centerList"]
        self.stdList=checkpoint["stdList"]
        self.dataset=dataset
        self.test_hunxiao=[[],[],[],[],[],[],[],[],[],[]]
        
        self.test_criterion=nn.L1Loss()
        self.test_criterion.requires_grad_ = False

    def mul_test_init(self,checkpointList,dataset):
        self.loss=[]
        for i in range(len(self.modelList)):
            self.modelList[i].load_time_checkpoint(checkpointList[i//3]["modelList"][i%3])
            self.modelList[i].extractor.eval()
            self.modelList[i].time_extractor.eval()
            self.modelList[i].regressor.eval()
            sub_loss=checkpointList[i//3]['loss']['mae'+str(i%3)]
            self.loss.append(sub_loss)

        tmp=[]
        for i in range(len(self.modelList)):
            tmp.append(math.log2(-1/0.2*self.loss[i]+2))
        self.loss=tmp
        tmp=[]
        for i in range(len(self.modelList)):
            tmp.append(self.loss[i]/sum(self.loss)*len(self.modelList))
        self.loss=tmp

        self.backbone.load_time_checkpoint(checkpointList[0]["backbone"])
        self.backbone.extractor.eval()
        self.backbone.time_extractor.eval()
        self.backbone.regressor.eval()

        self.centerList=[]
        self.stdList=[]
        for i in range(len(self.modelList)//3):
            self.centerList+=checkpointList[i]["centerList"]
            self.stdList+=checkpointList[i]["stdList"]
        self.dataset=dataset
        self.test_hunxiao=[[],[],[],[],[],[],[],[],[],[]]
        
        self.test_criterion=nn.L1Loss()
        self.test_criterion.requires_grad_ = False

    def test(self):
        bar = tqdm(total=len(self.dataset.test_dataloader), desc=f"test")
        cnt=0
        l1_loss = 0.0
        with open("/hdd/sdd/lzq/DLMM_new/model/loss.csv","w",newline='') as csvfile: 
            writer = csv.writer(csvfile)
            #writer.writerow(["groundtruth","predict","experts1","experts2","experts3","experts_all","weights1","weights2","weights3","weights_all"])
            with torch.no_grad():
                for data in self.dataset.test_dataloader:
                    if (data['xs'][0][0]).size()[0]<10:
                        continue
                    outputs_list=[]
                    fea_list=[]
                    for i in range(len(self.modelList)):
                        sub_outputs,y,fea=self.train_forward(data,is_train=False,model=self.modelList[i])
                        outputs_list.append(sub_outputs)
                        fea_list.append(fea.squeeze(0))
                    weights=self.whichCluster(fea_list)
                    #weights=self.whichCluster(data['xs'][self.modal])
                    outputs=0
                    for i in range(len(self.modelList)):
                        outputs+=weights[i]*self.loss[i]*outputs_list[i]

                    #print("groundtruth:",float(y),"  predict:",float(outputs),"  experts:",[float(i) for i in outputs_list],"  weights:",[float(i) for i in weights])
                    writer.writerow([float(y),float(outputs)]+[float(i) for i in outputs_list]+[float(i) for i in weights])

                    l1_loss_sub = self.test_criterion(outputs, y).item()

                    l1_loss +=l1_loss_sub
                    self.test_hunxiao[int(y//0.1)].append(l1_loss_sub)
                    cnt+=1
                    bar.set_postfix(**{'mae': l1_loss / cnt})
                    bar.update(1)
        bar.close()

        tmp=[]
        for hunxiao in self.test_hunxiao:
            if len(hunxiao):
                tmp.append(sum(hunxiao)/len(hunxiao))
        self.test_hunxiao=tmp

        print(l1_loss / cnt,self.test_hunxiao)

    def train(self,EPOCH,savepath):
        plt_loss_list=[[],[],[],[],[]]
        for epoch in range(EPOCH):
            for i in range(len(self.modelList)):
                self.modelList[i].extractor.eval()
                self.modelList[i].time_extractor.train()
                self.modelList[i].regressor.train()

            bar = tqdm(total=len(self.dataset.train_dataloader), desc=f"train epoch {epoch}")
            sum_loss_list=[0,0,0,0]
            l1_loss_list = [0,0,0,0]
            cnt_list=[0,0,0,0]
            fea_list=[[],[],[],[]]
            for data in self.dataset.train_dataloader:
                if (data['xs'][0][0]).size()[0]<10:
                    continue
                
                model_id=self.space_path[data['xs'][-1][0]]
                self.optimizer.zero_grad()
                outputs_model_id,y,fea_model_id=self.train_forward(data,is_train=False,model=self.modelList[model_id])
                #outputs_model_total,y,fea_model_total=self.train_forward(data,is_train=False,model=self.modelList[-1])
                loss_model_id = self.train_criterion(outputs_model_id, y)
                #loss_model_total=self.train_criterion(outputs_model_total, y)
                loss=loss_model_id
                loss.backward()
                self.optimizer.step()
                
                fea_list[model_id].append(fea_model_id.squeeze(0))
                sum_loss_list[model_id] += loss_model_id.item()
                l1_loss_list[model_id] += self.test_criterion(outputs_model_id, y).item()
                cnt_list[model_id]+=1

                # fea_list[-1].append(fea_model_total.squeeze(0))
                # sum_loss_list[-1] += loss_model_total.item()
                # l1_loss_list[-1]+=self.test_criterion(outputs_model_total, y).item()
                # cnt_list[-1]+=1

                # if epoch==0:
                #     fea_list[model_id].append(self.backbone_forward(data['xs'][self.modal]).squeeze(0))

                showdic={}
                for i in range(len(self.modelList)):
                    #showdic['loss'+str(i)]=sum_loss_list[i]/cnt_list[i] if cnt_list[i] else -1
                    showdic['mae'+str(i)]=l1_loss_list[i]/cnt_list[i] if cnt_list[i] else -1
                
                bar.set_postfix(**showdic)
                bar.update(1)
                
            bar.close()

            for i in range(len(self.modelList)):
                self.modelList[i].extractor.eval()
                self.modelList[i].time_extractor.eval()
                self.modelList[i].regressor.eval()
                
                plt_loss_list[i].append(l1_loss_list[i]/cnt_list[i])

                if 1:
                    inf=1e-9
                    center_total=torch.stack(fea_list[i]).mean(axis=0,keepdim=False)
                    std_total=torch.std(torch.stack(fea_list[i]),axis=0)+inf
                    self.centerList.append(center_total)
                    self.stdList.append(std_total)

            bar = tqdm(total=len(self.dataset.test_dataloader), desc=f"test epoch {epoch}")
            cnt=0
            l1_loss = 0.0
            with torch.no_grad():
                for data in self.dataset.test_dataloader:
                    if (data['xs'][0][0]).size()[0]<10:
                        continue
                    outputs_list=[]
                    fea_list=[]
                    for i in range(len(self.modelList)):
                        sub_outputs,y,fea=self.train_forward(data,is_train=False,model=self.modelList[i])
                        outputs_list.append(sub_outputs)
                        fea_list.append(fea.squeeze(0))
                    weights=self.whichCluster(fea_list)
                    #weights=self.whichCluster(data['xs'][self.modal])
                    outputs=0
                    for i in range(len(self.modelList)):
                        outputs+=weights[i]*outputs_list[i]
                    l1_loss_sub = self.test_criterion(outputs, y).item()
                    l1_loss +=l1_loss_sub
                    cnt+=1
                    self.test_hunxiao[int(y//0.1)].append(l1_loss_sub)

                    bar.set_postfix(**{'mae': l1_loss / cnt})
                    bar.update(1)
            bar.close()

            testloss=l1_loss / cnt
            plt_loss_list[-1].append(testloss)
            tmp=[]
            for hunxiao in self.test_hunxiao:
                if len(hunxiao):
                    tmp.append(sum(hunxiao)/len(hunxiao))
            self.test_hunxiao=tmp
            print(self.test_hunxiao)
            
            if testloss <= self.testloss_best:
                self.testloss_best = testloss
                save_modelList=[]
                for i in range(len(self.modelList)):
                    sub_save_modelList={}
                    sub_save_modelList['extractor']= self.modelList[i].extractor.state_dict()
                    sub_save_modelList['time_extractor']= self.modelList[i].time_extractor.state_dict()
                    sub_save_modelList['regressor']= self.modelList[i].regressor.state_dict()
                    save_modelList.append(sub_save_modelList)
                save_backbone={}
                save_backbone['extractor']= self.backbone.extractor.state_dict()
                save_backbone['time_extractor']= self.backbone.time_extractor.state_dict()
                save_backbone['regressor']= self.backbone.regressor.state_dict()

                state = {
                'modelList': save_modelList,
                'backbone': save_backbone,
                'loss':showdic,
                'acc': self.testloss_best,
                'test_hunxiao':self.test_hunxiao,
                "centerList":self.centerList,
                "stdList":self.stdList,
                'space_path':self.space_path
                }
                torch.save(state, savepath)

            self.test_hunxiao=[[],[],[],[],[],[],[],[],[],[]]
            self.centerList=[]
            self.stdList=[]
            print('  [Test] mae: %.03f  [Best] mae: %.03f'% (testloss,self.testloss_best))
        
            self.loss_visualize(epoch,plt_loss_list)
