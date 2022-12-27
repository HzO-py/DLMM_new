import csv
import random
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
from time import time
from scipy.signal import savgol_filter

class DataSet():
    def __init__(self,batch_size,TRAIN_RIO,DATA_PATHS,modal,is_time,collate_fn,pic_size):
        train_dataset=AllDataset(1,TRAIN_RIO,DATA_PATHS,modal,is_time,pic_size,0)
        val_dataset=AllDataset(0,TRAIN_RIO,DATA_PATHS,modal,is_time,pic_size,0)
        test_dataset=AllDataset(0,TRAIN_RIO,DATA_PATHS,modal,is_time,pic_size,1)
        train_dataset.get_y_label()
        val_dataset.get_y_label()
        test_dataset.get_y_label()
        if collate_fn is None:
            self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
            self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size,shuffle=False)
            self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)
        else:
            self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,collate_fn=collate_fn)
            self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size,shuffle=False,collate_fn=collate_fn)
            self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False,collate_fn=collate_fn)

class SingleModel():
    def __init__(self,extractor,time_extractor:Time_SelfAttention,regressor:Regressor,modal,prototype=None,cluster=None):
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
        outputs=F.relu(outputs)
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
        # if not is_train:
        #     x=x[0]
        for imgs in x:
            imgs=imgs.cuda()
            with torch.no_grad():
                _,fea,_=self.extractor(imgs)
            features.append(fea)
        features=torch.stack(features)
        outputs,lstm_output=self.time_extractor(features)
        if not is_dbscan:
            if is_selfatt:
                outputs=F.relu(self.regressor(outputs))
            else:
                outputs=F.relu(self.regressor(lstm_output))
        return outputs,y,xs[-1]

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
        #self.test_hunxiao=[[],[],[],[],[],[],[],[],[],[]]
        cnt=0
        sum_loss = 0.0
        bar = tqdm(total=len(self.dataset.test_dataloader), desc=f"test")
        with torch.no_grad():
            for data in self.dataset.test_dataloader:
                outputs,y=self.extractor_forward(data,is_train=False)
                loss = criterion(outputs, y).detach().data
                sum_loss+=loss
                cnt+=1
                #self.test_hunxiao[int(y//0.1)].append(loss)
                bar.set_postfix(**{'mae': '{:.3f}'.format(sum_loss / cnt)})
                bar.update(1)
            bar.close()

            # tmp=[]
            # for hunxiao in self.test_hunxiao:
            #     if len(hunxiao):
            #         tmp.append(float(torch.sum(hunxiao)/len(hunxiao)))
            # self.test_hunxiao=tmp
        sum_loss=float(sum_loss.cpu())
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
            cnt=0
            for data in self.dataset.train_dataloader:
                self.optimizer.zero_grad()

                outputs,y=self.extractor_forward(data,is_train=True)
                loss = self.train_criterion(outputs, y)
                

                loss.backward()
                self.optimizer.step()
                sum_loss += loss.detach().data

                l1_loss += self.test_criterion(outputs, y).detach().data
                cnt+=1
                bar.set_postfix(**{'Loss': '{:.3f}'.format(sum_loss / cnt), 'mae': '{:.3f}'.format(l1_loss / cnt)})
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
            # print(['{:.3f}'.format(elem) for elem in self.test_hunxiao])

    def time_extractor_train(self,EPOCH,savepath,is_selfatt):
        self.extractor.eval()
        for epoch in range(EPOCH):
            bar = tqdm(total=len(self.dataset.train_dataloader), desc=f"train epoch {epoch}")
            for net in self.train_nets:
                net.train()
            sum_loss=0.0
            l1_loss = 0.0
            cnt=0
            for data in self.dataset.train_dataloader:
                if epoch==0:
                    break
                self.optimizer.zero_grad()
                outputs,y,_=self.time_extractor_forward(data,is_train=True,is_selfatt=is_selfatt)
                loss = self.train_criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                sum_loss += loss.detach().data
                l1_loss += self.test_criterion(outputs, y).detach().data
                cnt+=1
                bar.set_postfix(**{'Loss': '{:.3f}'.format(sum_loss / cnt), 'mae': '{:.3f}'.format(l1_loss / cnt)})
                bar.update(1)
            bar.close()

            for net in self.train_nets:
                net.eval()
            cnt=0
            l1_loss = 0.0
            self.test_hunxiao=[[],[],[],[],[],[],[],[],[],[]]
            bar = tqdm(total=len(self.dataset.test_dataloader), desc=f"test epoch {epoch}")
            with torch.no_grad():
                for data in self.dataset.test_dataloader:
                    outputs,y,_=self.time_extractor_forward(data,is_train=False,is_selfatt=is_selfatt)
                    l1_loss_sub = self.test_criterion(outputs, y).detach().data
                    l1_loss +=l1_loss_sub
                    cnt+=1
                    self.test_hunxiao[int(y//0.1)].append(l1_loss_sub)
                    bar.set_postfix(**{'mae': '{:.3f}'.format(l1_loss / cnt)})
                    bar.update(1)
            bar.close()
            testloss=l1_loss / cnt
            testloss=float(testloss.cpu())
            tmp=[]
            for hunxiao in self.test_hunxiao:
                if len(hunxiao):
                    tmp.append(float(torch.sum(torch.stack(hunxiao))/len(hunxiao)))
            self.test_hunxiao=tmp
            print(self.test_hunxiao)
            
            if testloss < self.testloss_best:
                self.testloss_best = testloss
                state = {
                'extractor': self.extractor.state_dict(),
                'time_extractor': self.time_extractor.state_dict(),
                'regressor': self.regressor.state_dict(),
                'acc': self.testloss_best,
                'test_hunxiao': self.test_hunxiao
            }
                torch.save(state, savepath)
                if epoch<=10:
                    torch.save(state, savepath[-3]+'_epoch10.t7')
            
            print('  [Test] mae: %.03f  [Best] mae: %.03f'% (testloss,self.testloss_best))
            print(['{:.3f}'.format(elem) for elem in self.test_hunxiao])

    def loss_visualize(self,epoch, plt_loss_list,savepath):
        epochs=[i for i in range(epoch+1)]
        plt.style.use("ggplot")
        plt.figure()
        plt.title("Epoch_Loss")
        plt.plot(epochs, plt_loss_list[0], label='y_std', color='r', linestyle='-')
        plt.plot(epochs, plt_loss_list[1], label='center_std/2.0', color='g', linestyle='-')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(savepath,'loss.jpg'))
        plt.close()

    def feature_space(self,is_selfatt,savepath,pre_space_path,pre_model_id,model_id,sample_threshold,score_threshold,pre_score,cluster_num):
        if not os.path.exists(savepath):
            os.makedirs(savepath)
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
                    if len(pre_space_path)>0 and pre_space_path[data['xs'][-1][0]]!=pre_model_id:
                        continue
                    outputs,y,path=self.time_extractor_forward(data,is_train=False,is_selfatt=is_selfatt,is_dbscan=True)
                    space_fea.append(outputs.cpu().squeeze(0))
                    space_y.append(y.cpu().squeeze(0))
                    space_path.append(path)
                    bar.update(1)
                bar.close()
            space_fea=torch.stack(space_fea)
            space_y=torch.stack(space_y)
            torch.save(space_fea,os.path.join(savepath,'space_fea.pt'))
            torch.save(space_y,os.path.join(savepath,'space_y.pt'))
            np.save(os.path.join(savepath,'space_path.npy'),np.array(space_path))
        return self.cluster_space(space_fea.cuda(),space_y.cuda(),space_path,savepath,model_id=model_id,sample_threshold=sample_threshold,score_threshold=score_threshold,pre_score=pre_score,cluster_num=cluster_num)
        

    def cluster_space(self,space_fea,space_y,space_path,savepath,model_id,sample_threshold,score_threshold,pre_score,cluster_num):
        if os.path.exists(os.path.join(savepath,'cluster_centerList.npy')):
            space_path=np.load(os.path.join(savepath,'cluster_space_path.npy'), allow_pickle='TRUE')
            space_path=space_path.item()
            centerList=np.load(os.path.join(savepath,'cluster_centerList.npy'))
            stdList=np.load(os.path.join(savepath,'cluster_stdList.npy'))
            num=1 if centerList.shape[0]<2 else cluster_num

        else:
            for net in self.train_nets:
                net.train()
            for net in self.forward_nets:
                net.eval()
            
            rand_id_list=random.sample(range(0,space_fea.shape[0]),cluster_num)
            rand_fea_list=[space_fea[x] for x in rand_id_list]

            with torch.no_grad():
                new_para=self.cluster.state_dict()
                new_para['fc1.weight']=torch.stack(rand_fea_list).cuda()
                self.cluster.load_state_dict(new_para)

            num=cluster_num
            EPOCH_SIZE=1000
            inf=1e-9
            plt_loss_list=[[],[]]
            center_total=[0,0,0]
            std_total=[inf,inf,inf]
            loss_best=1e5
            group_best={}
            cluster_labels_best=[0]*space_fea.shape[0]
            bar = tqdm(total=EPOCH_SIZE, desc=f"cluster_space")
            for epoch in range(EPOCH_SIZE):
                self.optimizer.zero_grad()
                loss_fea=[]
                loss_y=[]
                y_list=[[],[],[]]
                fea_list=[[],[],[]]
                cluster_labels=[0]*space_fea.shape[0]
                group={}

                for i in range(space_fea.shape[0]):
                    fea,center=self.cluster(space_fea[i])
                    gussian_distance=[]
                    for k in range(num):
                        eu_distance = -1*torch.norm(fea - center[0][k]) #计算第k（k=0或1）个中心center[0][k]和当前第i个特征fea的负的欧式距离
                        gussian_distance.append(float(torch.exp(eu_distance))) #gussian_distance越大，距离越小
                    maxn_k=np.argmax(np.array(gussian_distance))
                    #maxn_k=0 if gussian_distance[0]>gussian_distance[1] else 1 #看哪个中心离fea的距离近，记为maxn_k
                    #subloss=-torch.log(gussian_distance[maxn_k])-torch.log(1-gussian_distance[1-maxn_k]) #拉近离fea近的中心maxn_k，推远离fea远的中心1-maxn_k
                    #loss_fea.append(subloss)
                    
                    cluster_labels[i]=maxn_k
                    group[space_path[i][0]]=maxn_k+model_id
                    y_list[maxn_k].append(space_y[i])
                    fea_list[maxn_k].append(fea)

                num=1 if min([len(y) for y in y_list])<sample_threshold else cluster_num

                if num>1:
                    for k in range(num):
                        dists=[]
                        for i in range(len(fea_list[k])):
                            dist_k=-torch.log(torch.exp(-torch.norm(fea_list[k][i]-center[0][k])))
                            for kk in range(num):
                                if kk!=k:
                                    dist_k+=-torch.log(1.0-torch.exp(-torch.norm(fea_list[k][i]-center[0][kk])))
                            dists.append(dist_k)
                        loss_y.append(torch.stack(dists).mean(axis=0,keepdim=False))

                        

                    #loss_fea=torch.stack(loss_fea).mean(axis=0,keepdim=False)
                    loss_y=torch.stack(loss_y).mean(axis=0,keepdim=False)
                    loss=loss_y
                    loss.backward()
                    self.optimizer.step()

                    if loss.item()<loss_best:
                        for k in range(num):  
                            loss_best=loss.item()
                            center_total[k]=torch.stack(fea_list[k]).mean(axis=0,keepdim=False)
                            std_total[k]=torch.std(torch.stack(fea_list[k]),axis=0)+inf
                        group_best=group
                        cluster_labels_best=cluster_labels


                    # y_std=0
                    # center_std=0
                    # inf=1e-9
                    # for k in range(3):
                    #     center=torch.stack(fea_list[k]).mean(axis=0,keepdim=False)
                    #     eu_distance = -1*torch.norm(center-space_fea.mean(axis=0,keepdim=False))
                    #     gussian_distance=torch.exp(eu_distance)
                    #     center_std+=float(1-gussian_distance)+inf

                    #     y_std+=float(torch.std(torch.stack(y_list[k]),axis=0))

                    # # score=y_std/center_std
                    # # score_percent=(pre_score-score)/pre_score

                    # plt_loss_list[0].append(y_std)
                    # plt_loss_list[1].append(center_std/3.0)
                    # self.loss_visualize(epoch,plt_loss_list,savepath)
                
                else:
                    break

                bar.set_postfix(**{'Loss': loss.cpu().item()})
                bar.update(1)
            bar.close()
            
            num=cluster_num
            space_path=group_best
            # if num==1:
            #     centerList=[space_fea.mean(axis=0,keepdim=False).unsqueeze(0)]
            #     for key in space_path.keys():
            #         space_path[key]=model_id

            space_fea=space_fea.cpu().detach().numpy()
            space_y=space_y.cpu().detach().numpy()
            cluster_labels=np.array(cluster_labels_best)
            centerList=torch.stack(center_total).cpu().detach().numpy()
            stdList=torch.stack(std_total).cpu().detach().numpy()

            if num>1:
                self.tsne_space(space_fea,space_y,cluster_labels,num,savepath)
                            
                
            np.save(os.path.join(savepath,'cluster_space_path.npy'),space_path)
            np.save(os.path.join(savepath,'cluster_centerList.npy'),centerList)
            np.save(os.path.join(savepath,'cluster_stdList.npy'),stdList)
            print(loss_best)
        
        return space_path,num,centerList,stdList

    def tsne_space(self,space_fea,space_y,labels,num,savepath):
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
        for i in range(num):
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
    def __init__(self,modelList:List[SingleModel]):
        self.modelList=modelList
        for i in range(len(self.modelList)):
            self.modelList[i].extractor.cuda()
            self.modelList[i].time_extractor.cuda()
            self.modelList[i].regressor.cuda()

    def load_checkpoint(self,checkpointList,space_path=None,centerList=None,stdList=None):
        for i in range(len(self.modelList)):
            self.modelList[i].load_time_checkpoint(checkpointList[i])
        self.centerList=[]
        self.stdList=[]
        self.space_path=[0,0,1,1,2,2,2,2,2]
        if space_path:
            self.space_path=space_path
        if centerList is not None:
            self.centerList=torch.from_numpy(centerList).cuda() if type(centerList)==np.ndarray else centerList.cuda()
        if stdList is not None:
            self.stdList=torch.from_numpy(stdList).cuda() if type(stdList)==np.ndarray else stdList.cuda()
        
    def train_init(self,dataset,LR,WEIGHT_DELAY):
        self.dataset=dataset
        nets=[]
        for i in range(len(self.modelList)):
            nets+=[self.modelList[i].time_extractor,self.modelList[i].regressor]
        opt=optim.Adam(chain(*[net.parameters() for net in nets]), lr=LR,weight_decay=WEIGHT_DELAY)
        self.optimizer=opt
        
        self.train_criterion=nn.MSELoss()
        self.test_criterion=nn.L1Loss()
        self.test_criterion.requires_grad_ = False
        self.testloss_best=1e5
        self.test_hunxiao=[[],[],[],[],[],[],[],[],[],[]]

    def train_forward(self,data,is_train,model:SingleModel):
        #backbone=self.modelList[0]
        xs,y=data.values()                
        y = y.to(torch.float32).unsqueeze(1)
        # if is_train:
        #     y = y + 0.1*torch.randn(y.size()[0],1)
        y[y<0]=0
        y = y.cuda()

        features = []
        with torch.no_grad():
            for imgs in xs[model.modal]:
                imgs=imgs.cuda()
                _,fea,_=model.extractor(imgs)
                features.append(fea)
            features=torch.stack(features)
        time_outputs,_=model.time_extractor(features)
        outputs=F.relu(model.regressor(time_outputs))

        return outputs,y,time_outputs

    def GuassianDist(self,x,y,s):
        eu_distance = -1*torch.norm((x-y)/s) #负的欧式距离
        gussian_distance = torch.exp(eu_distance)
        return gussian_distance

    def tuilaLoss(self,x,k):
        inf=1e-9
        dis_list=[]
        #x_fea=self.backbone_forward(x)
        for i in range(len(self.modelList)):
            dis=self.GuassianDist(x,self.centerList[i],self.stdList[i])
            dis_list.append(dis)
        dis_list=torch.stack(dis_list)
        loss=-(dis_list[k]/(torch.sum(dis_list)+inf))
        return loss

    def whichCluster(self,x,top1=False):
        inf=1e-9
        dis_list=[]
        #x_fea=self.backbone_forward(x)
        for i in range(len(self.modelList)):
            dis=self.GuassianDist(x[i],self.centerList[i],self.stdList[i])
            dis_list.append(dis)
        weights=[]
        for i in range(len(self.modelList)):
            weights.append(dis_list[i]/(sum(dis_list)+inf))

        if top1:
            max_id=weights.index(max(weights))
            return max_id
            for i in range(len(self.modelList)):
                weights[i]=1 if i==max_id else 0

        return weights
    
    def loss_visualize(self,epoch, plt_loss_list):
        epochs=[i for i in range(epoch+1)]
        plt.style.use("ggplot")
        plt.figure()
        plt.title("Epoch_Loss")
        plt.plot(epochs, plt_loss_list[0], label='mlp_loss', color='r', linestyle='-')
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

        self.centerList=checkpoint["centerList"]
        self.stdList=checkpoint["stdList"]
        self.dataset=dataset
        self.test_hunxiao=[[],[],[],[],[],[],[],[],[],[]]
        
        self.test_criterion=nn.L1Loss()
        self.test_criterion.requires_grad_ = False

    def mul_test_init(self,checkpointList,dataset):
        inf=1e-9
        for i in range(len(self.modelList)):
            self.modelList[i].load_time_checkpoint(checkpointList[i//3]["modelList"][i%3])
            self.modelList[i].extractor.eval()
            self.modelList[i].time_extractor.eval()
            self.modelList[i].regressor.eval()

        for i in range(len(checkpointList)):
            print(checkpointList[i]["acc"])
            print(checkpointList[i]["test_hunxiao"])

        self.centerList=[]
        self.stdList=[]
        for i in range(len(self.modelList)//3):
            self.centerList+=checkpointList[i]["centerList"]
            self.stdList+=checkpointList[i]["stdList"]
        self.dataset=dataset
        self.test_hunxiao=[[],[],[],[],[],[],[],[],[],[]]
        
        self.test_criterion=nn.L1Loss()
        self.test_criterion.requires_grad_ = False

        with torch.no_grad():
            bar = tqdm(total=len(self.dataset.train_dataloader), desc=f"train")
            train_loss=[[inf]*10,[inf]*10,[inf]*10,[inf]*10,[inf]*10,[inf]*10,[inf]*10,[inf]*10,[inf]*10]
            train_loss_cnt=[[0]*10,[0]*10,[0]*10,[0]*10,[0]*10,[0]*10,[0]*10,[0]*10,[0]*10]
            for data in self.dataset.train_dataloader:
                if (data['xs'][0][0]).size()[0]<10:
                    continue

                
                for i in range(len(checkpointList)):
                    model_id=checkpointList[i]['space_path'][data['xs'][-1][0]]+i*3
                    outputs,y,fea_model_id=self.train_forward(data,is_train=False,model=self.modelList[model_id])
                    l1_loss_sub = self.test_criterion(outputs, y).item()
                    train_loss[model_id][int(float(outputs.cpu())//0.1)]+=l1_loss_sub
                    train_loss_cnt[model_id][int(float(outputs.cpu())//0.1)]+=1
                bar.update(1)
            bar.close()
            for i in range(len(self.modelList)):
                for j in range(10):
                    train_loss[i][j]=1e5 if train_loss_cnt[i][j]==0 else train_loss[i][j]/train_loss_cnt[i][j]
        self.train_loss=train_loss

    def weights_fusion(self,w1,w2):
        w=[]
        for i in range(len(w1)):
            w.append(w1[i]*w2[i])
        return [x/sum(w) for x in w]    

    def test(self):        
        cnt=0
        l1_loss = 0.0
        with open("/hdd/sdd/lzq/DLMM_new/model/loss2.csv","w",newline='') as csvfile: 
            writer = csv.writer(csvfile)
            #writer.writerow(["groundtruth","predict","experts1","experts2","experts3","experts_all","weights1","weights2","weights3","weights_all"])
            with torch.no_grad():
                bar = tqdm(total=len(self.dataset.test_dataloader), desc=f"test")
                for data in self.dataset.test_dataloader:
                    if (data['xs'][0][0]).size()[0]<10:
                        continue
                    outputs_list=[]
                    fea_list=[]
                    loss_weights=[]
                    for i in range(len(self.modelList)):
                        sub_outputs,y,fea=self.train_forward(data,is_train=False,model=self.modelList[i])
                        outputs_list.append(sub_outputs)
                        fea_list.append(fea.squeeze(0))
                        loss_weights.append(1.0/self.train_loss[i][int(float(sub_outputs.cpu())//0.1)])
                    weights=self.whichCluster(fea_list)
                    loss_weights=[x/sum(loss_weights) for x in loss_weights]
                    weights=self.weights_fusion(weights,loss_weights)
                    #weights=self.whichCluster(data['xs'][self.modal])
                    outputs=0
                    for i in range(len(self.modelList)):
                        outputs+=weights[i]*outputs_list[i]

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
        plt_loss_list=[[],[],[],[],[],[],[],[],[]]
        inf=1e-9
        for epoch in range(EPOCH):
        
            for i in range(len(self.modelList)):
                self.modelList[i].extractor.eval()
                self.modelList[i].time_extractor.train()
                self.modelList[i].regressor.train()

            bar = tqdm(total=len(self.dataset.train_dataloader), desc=f"train epoch {epoch}")
            sum_loss_list=[0,0,0,0,0,0,0,0]
            l1_loss_list = [0,0,0,0,0,0,0,0]
            cnt_list=[0,0,0,0,0,0,0,0]
            fea_list=[[],[],[],[],[],[],[],[]]
            y_list=[[],[],[],[],[],[],[],[]]

            labels=[]
            space_fea=[]
            space_y=[]

            train_loss=[[inf]*10,[inf]*10,[inf]*10]
            train_loss_cnt=[[0]*10,[0]*10,[0]*10]
            for data in self.dataset.train_dataloader:
                if (data['xs'][0][0]).size()[0]<10:
                    continue
                
                model_id=self.space_path[data['xs'][-1][0]]
                #model_id=self.space_path[int(float(data['y'][0])/0.1)]

                labels.append(model_id)

                self.optimizer.zero_grad()
                outputs_model_id,y,fea_model_id=self.train_forward(data,is_train=False,model=self.modelList[model_id])

                space_fea.append(fea_model_id.cpu().squeeze(0))
                space_y.append(y.cpu().squeeze(0))

                #outputs_model_total,y,fea_model_total=self.train_forward(data,is_train=False,model=self.modelList[-1])
                loss_model_id = self.train_criterion(outputs_model_id, y)
                #loss_model_total=self.train_criterion(outputs_model_total, y)
                
                tuilaLoss=self.tuilaLoss(fea_model_id,model_id) if epoch>0 else 0
                loss=loss_model_id+(tuilaLoss+1.0)*0.01

                loss.backward()
                self.optimizer.step()
                
                fea_list[model_id].append(fea_model_id.squeeze(0))
                y_list[model_id].append(y)
                sum_loss_list[model_id] += loss_model_id.detach().data
                l1_loss_sub = self.test_criterion(outputs_model_id, y).item()
                l1_loss_list[model_id]+=l1_loss_sub
                cnt_list[model_id]+=1

                outputs_model_id=outputs_model_id.item()
                tmp_id=int(float(outputs_model_id)//0.1)
                if tmp_id>=0 and tmp_id<10:
                    train_loss[model_id][tmp_id]+=float(l1_loss_sub)
                    train_loss_cnt[model_id][tmp_id]+=1

                # fea_list[-1].append(fea_model_total.squeeze(0))
                # sum_loss_list[-1] += loss_model_total.item()
                # l1_loss_list[-1]+=self.test_criterion(outputs_model_total, y).item()
                # cnt_list[-1]+=1

                # if epoch==0:
                #     fea_list[model_id].append(self.backbone_forward(data['xs'][self.modal]).squeeze(0))

                showdic={}
                for i in range(len(self.modelList)):
                    #showdic['loss'+str(i)]=sum_loss_list[i]/cnt_list[i] if cnt_list[i] else -1
                    showdic['mae'+str(i)]='{:.3f}'.format(l1_loss_list[i]/cnt_list[i]) if cnt_list[i] else -1
                
                bar.set_postfix(**showdic)
                bar.update(1)
                
            bar.close()

            for i in range(len(self.modelList)):
                for j in range(10):
                    train_loss[i][j]=1e5 if train_loss_cnt[i][j]==0 else train_loss[i][j]/train_loss_cnt[i][j]
            self.train_loss=train_loss
            # space_y=torch.stack(space_y).cpu().detach().numpy()
            # space_fea=torch.stack(space_fea).cpu().detach().numpy()
            # labels=np.array(labels)
            # plt_save_path='\hdd\sdd\lzq\DLMM_new\model\cluster\\experts_zd_bio_3f_0G_aftertrain.t7'
            # if not os.path.exists(plt_save_path):
            #     os.makedirs(plt_save_path)
            # self.modelList[0].tsne_space(space_fea,space_y,labels,3,plt_save_path)
            # return

            with torch.no_grad():
                for i in range(len(self.modelList)):
                    plt_loss_list[i].append(l1_loss_list[i]/cnt_list[i])

                    center_total=torch.stack(fea_list[i]).mean(axis=0,keepdim=False)
                    std_total=torch.std(torch.stack(fea_list[i]),axis=0)+inf
                    self.centerList[i]=center_total
                    self.stdList[i]=std_total

                    self.modelList[i].extractor.eval()
                    self.modelList[i].time_extractor.eval()
                    self.modelList[i].regressor.eval()

            # with torch.no_grad():
            #     bar = tqdm(total=len(self.dataset.val_dataloader), desc=f"val epoch {epoch}")
            #     cnt=0
            #     l1_loss_list = [0,0,0,0,0,0,0,0]
            #     cnt_list=[0,0,0,0,0,0,0,0]
                
            #     for data in self.dataset.val_dataloader:
            #         if (data['xs'][0][0]).size()[0]<10:
            #             continue
            #         outputs_list=[]
            #         fea_list=[]
            #         for i in range(len(self.modelList)):
            #             sub_outputs,y,fea=self.train_forward(data,is_train=False,model=self.modelList[i])
            #             outputs_list.append(sub_outputs)
            #             fea_list.append(fea.squeeze(0))
            #         max_id=self.whichCluster(fea_list,True)
            #         outputs=outputs_list[max_id]
            #         l1_loss_list[max_id]+=self.test_criterion(outputs, y).detach().data
            #         cnt_list[max_id]+=1

            #         showdic={}
            #         for i in range(len(self.modelList)):
            #             showdic['mae'+str(i)]='{:.3f}'.format(l1_loss_list[i]/cnt_list[i]) if cnt_list[i] else -1
            #         self.loss=[float(x) for x in showdic.values()]
            #         bar.set_postfix(**showdic)
            #         bar.update(1)

            #         # fea_list=torch.stack(fea_list).cpu().detach().numpy()
            #         # tsne_fea=TSNE().fit_transform(np.concatenate((space_fea,fea_list),axis=0))
            #         # plt.scatter(tsne_fea[:-3,0],tsne_fea[:-3,1],c=space_y//0.1*0.1,cmap="coolwarm")
            #         # plt.scatter(tsne_fea[-3,0],tsne_fea[-3,1],c='r',s=50)
            #         # plt.scatter(tsne_fea[-2,0],tsne_fea[-2,1],c='g',s=50)
            #         # plt.scatter(tsne_fea[-1,0],tsne_fea[-1,1],c='b',s=50)
            #         # plt.show()
            #         # plt.close()


            # bar.close()
                        
            with torch.no_grad():
                bar = tqdm(total=len(self.dataset.test_dataloader), desc=f"test epoch {epoch}")
                cnt=0
                l1_loss = 0.0
                
                for data in self.dataset.test_dataloader:
                    if (data['xs'][0][0]).size()[0]<10:
                        continue
                    outputs_list=[]
                    fea_list=[]
                    loss_weights=[]
                    for i in range(len(self.modelList)):
                        sub_outputs,y,fea=self.train_forward(data,is_train=False,model=self.modelList[i])
                        outputs_list.append(sub_outputs)
                        fea_list.append(fea.squeeze(0))
                        loss_weights.append(1.0/self.train_loss[i][int(float(sub_outputs.cpu())//0.1)])
                    weights=self.whichCluster(fea_list)
                    loss_weights=[x/sum(loss_weights) for x in loss_weights]
                    weights=self.weights_fusion(weights,loss_weights)
                    #weights=self.whichCluster(data['xs'][self.modal])
                    outputs=0
                    for i in range(len(self.modelList)):
                        outputs+=weights[i]*outputs_list[i]
                    l1_loss_sub = self.test_criterion(outputs, y).detach().data
                    l1_loss +=l1_loss_sub
                    cnt+=1
                    self.test_hunxiao[int(y//0.1)].append(l1_loss_sub)

                    bar.set_postfix(**{'mae':  '{:.3f}'.format(l1_loss / cnt)})
                    bar.update(1)
            bar.close()

            testloss=l1_loss.item() / cnt
            plt_loss_list[-1].append(testloss)
            tmp=[]
            for hunxiao in self.test_hunxiao:
                if len(hunxiao):
                    tmp.append('{:.3f}'.format(sum(hunxiao)/len(hunxiao)))
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

                state = {
                'modelList': save_modelList,
                'loss':self.train_loss,
                'acc': self.testloss_best,
                'test_hunxiao':self.test_hunxiao,
                "centerList":self.centerList,
                "stdList":self.stdList,
                'space_path':self.space_path,
                'score':0
                }
                torch.save(state, savepath)

            self.test_hunxiao=[[],[],[],[],[],[],[],[],[],[]]
            print('  [Test] mae: %.03f  [Best] mae: %.03f'% (testloss,self.testloss_best))
        
            self.loss_visualize(epoch,plt_loss_list)

    def train_fusioner(self,EPOCH,savepath):
        plt_loss_list=[[],[],[],[],[],[],[],[],[]]
        for i in range(len(self.modelList)):
                self.modelList[i].extractor.eval()
                self.modelList[i].time_extractor.eval()
                self.modelList[i].regressor.eval()

        bar = tqdm(total=EPOCH, desc=f"train_fusioner")
        for epoch in range(EPOCH):
            self.fusioner.train()

            self.test_hunxiao=[[],[],[],[],[],[],[],[],[],[]]
            save_npy_train=[]
            y_list=[]

            if os.path.exists("\hdd\sdd\lzq\DLMM_new\model\cluster\\save_npy_train.pt"):
                save_npy_train=torch.load("\hdd\sdd\lzq\DLMM_new\model\cluster\\save_npy_train.pt").cuda()
                y_list=torch.load("\hdd\sdd\lzq\DLMM_new\model\cluster\\y_list.pt").cuda()
            else:
                for data in self.dataset.train_dataloader:
                    if (data['xs'][0][0]).size()[0]<10:
                        continue

                    outputs_list=[]
                    
                    with torch.no_grad():
                        for i in range(len(self.modelList)):
                            sub_outputs,y,fea=self.train_forward(data,is_train=False,model=self.modelList[i])
                            outputs_list.append(float(sub_outputs.squeeze(0)))
                            if i==0:
                                y_list.append(float(y.squeeze(0)))
                    outputs_list=torch.Tensor(outputs_list)
                    save_npy_train.append(outputs_list)
                    
                save_npy_train=torch.stack(save_npy_train).cuda()
                y_list=torch.Tensor(y_list).cuda()
                torch.save(save_npy_train,"\hdd\sdd\lzq\DLMM_new\model\cluster\\save_npy_train.pt")
                torch.save(y_list,"\hdd\sdd\lzq\DLMM_new\model\cluster\\y_list.pt")

            self.optimizer.zero_grad()    
            outputs=self.fusioner(save_npy_train).squeeze(-1)
            loss_model_id = self.train_criterion(outputs, y_list)
            loss=loss_model_id
            
            loss.backward()
            self.optimizer.step()

            plt_loss_list[0].append(loss.item())

            self.fusioner.eval()
            l1_loss = 0.0
            cnt=0
            bar = tqdm(total=len(self.dataset.test_dataloader), desc=f"test epoch {epoch}")
            with torch.no_grad():
                for data in self.dataset.test_dataloader:
                    if (data['xs'][0][0]).size()[0]<10:
                        continue
                    outputs_list=[]
                    for i in range(len(self.modelList)):
                        sub_outputs,y,fea=self.train_forward(data,is_train=False,model=self.modelList[i])
                        outputs_list.append(float(sub_outputs.squeeze(0)))
                    outputs_list=torch.Tensor(outputs_list).unsqueeze(0).cuda()
                    outputs=self.fusioner(outputs_list)
                    
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
                state = {
                'fusioner':self.fusioner,
                'acc': self.testloss_best,
                'test_hunxiao':self.test_hunxiao,
                }
                torch.save(state, savepath)

            print('  [Train] mae: %.03f  [Test] mae: %.03f  [Best] mae: %.03f'% (loss.item(),testloss,self.testloss_best))
            self.loss_visualize(epoch,plt_loss_list)
  