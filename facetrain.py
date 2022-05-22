# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 08:18:26 2020

@author: 98669
"""
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
from models import BILSTM, Prototype,Classifier,ResNet18,cnn1d,VGG,Regressor,TemporalConvNet
from loader import BioDataset,FaceDataset,VoiceDataset,FVDataset
import pdb
import matplotlib.pyplot as plt

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
parser = argparse.ArgumentParser(description='PyTorch Biovid PreTraining')
parser.add_argument('--yamlFile', default='config/face_config.yaml', help='yaml file') 
args = parser.parse_args()

cfg=getCfg(args.yamlFile)

os.environ["CUDA_VISIBLE_DEVICES"] = cfg["GPUID"]
EPOCH = cfg["EPOCH"]
SUB_EPOCH=cfg["SUB_EPOCH"]
pre_epoch = 0  
BATCH_SIZE = cfg["BATCH_SIZE"]
TCN_BATCH_SIZE=cfg["TCN_BATCH_SIZE"]
LR=cfg["LR"]
WEIGHT_DELAY=cfg["WEIGHT_DELAY"]
FACE_OR_VOICE=cfg["FACE_OR_VOICE"]

VGG_OR_RESNET=cfg["VGG_OR_RESNET"]
EXTRACT_NUM=cfg["EXTRACT_NUM"]
HIDDEN_NUM=cfg["HIDDEN_NUM"]
CLASS_NUM=cfg["CLASS_NUM"]

TCN_OR_LSTM=cfg["TCN_OR_LSTM"]
TCN_NUM=cfg["TCN_NUM"]
TCN_HIDDEN_NUM=cfg["TCN_HIDDEN_NUM"]

DATA_ROOT=cfg["DATA_ROOT"]
MODEL_ROOT=cfg["MODEL_ROOT"]
LOGS_ROOT=cfg["LOGS_ROOT"]
MODEL_NAME=cfg["MODEL_NAME"]
CHECKPOINT_NAME=cfg["CHECKPOINT_NAME"]

TRAIN_RIO=cfg["TRAIN_RIO"]
DATA_PATHS=cfg["DATA_PATHS"]
IS_POINT=cfg["IS_POINT"]


for i in range(len(DATA_PATHS)):
    modal='face'
    if not FACE_OR_VOICE:
        modal='face'
    else:
        modal='voice'
    DATA_PATHS[i][0]=os.path.join(DATA_PATHS[i][0],modal)



def tcnTest():
    checkpoint = torch.load(os.path.join(LOGS_ROOT, MODEL_NAME))
    print(checkpoint["acc"])
    resnet=torchvision.models.resnet18(pretrained=True)
    if FACE_OR_VOICE:
        resnet.conv1= nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
    net2=nn.Sequential(*list(resnet.children())[:-1])

    net2.load_state_dict(checkpoint['net2'])
    net2 = net2.cuda()
    net3 = Regressor(EXTRACT_NUM,HIDDEN_NUM)
    net3.load_state_dict(checkpoint['net3'])
    net3 = net3.cuda() 
    net4=TemporalConvNet(TCN_NUM,TCN_HIDDEN_NUM)
    net4.load_state_dict(checkpoint['net4'])
    net4 = net4.cuda()
    net5 = Regressor(HIDDEN_NUM,HIDDEN_NUM)
    net5.load_state_dict(checkpoint['net5'])
    net5 = net5.cuda()

    if FACE_OR_VOICE:
        test_datasets=[VoiceDataset(0,TRAIN_RIO,DATA_PATHS,is_person=1,tcn_num=TCN_NUM,person_test=1)]
    else:
        test_datasets=[FaceDataset(0,TRAIN_RIO,DATA_PATHS,is_person=1,tcn_num=TCN_NUM,person_test=1)]

    test_dataloaders=[]
    for test_dataset in test_datasets:
        test_dataloaders.append(DataLoader(test_dataset, batch_size=1,shuffle=False))
    for test_dataloader in test_dataloaders:
        sum_loss = 0.0
        cnt=0
        net2.eval()
        net3.eval()
        net4.eval()
        net5.eval()
        result=[]
        with torch.no_grad():
            
            for i,data in enumerate(test_dataloader):
                xss,npys,y=data.values()
                y=float(y)
                loss=[]
                
                for xs in xss:
                    feas=[]
                    for x in xs:
                        x = x.cuda()
                        x = net2(x)
                        x=x.squeeze(-1).squeeze(-1)
                        
                        _,fea = net3(x)
                        feas.append(fea)

                    feas=torch.stack(feas)
                    
                    feas = net4(feas).squeeze(1)
                    outputs,_=net5(feas)
                    outputs=outputs
                    #loss+=float(outputs.cpu())
                    #loss=max(loss,float(outputs.cpu()))
                    loss.append(float(outputs.cpu()))
                loss.sort()
                half=len(loss)//2
                loss=(loss[half]+loss[~half])/2
                #loss/=len(xss)
                cnt+=1
                #print(y,loss)
                sum_loss+=abs(loss-y)
                
                result.append([loss,y])
                    # 每训练1个batch打印一次loss和准确率
        result=sorted(result,key=lambda x:abs(x[0]-x[1]))           
        print(result)
        print('  [Test] Loss: %.03f'
                  % (sum_loss / cnt))

def fvTest():
    checkpoint = torch.load(os.path.join(LOGS_ROOT, MODEL_NAME))

    resnet_face=torchvision.models.resnet18(pretrained=True)
    net2_face=nn.Sequential(*list(resnet_face.children())[:-1])

    resnet_voice=torchvision.models.resnet18(pretrained=True)
    resnet_voice.conv1= nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
    net2_voice=nn.Sequential(*list(resnet_voice.children())[:-1])

    net3_face = Regressor(EXTRACT_NUM,HIDDEN_NUM)
    
    net3_voice = Regressor(EXTRACT_NUM,HIDDEN_NUM)

    
    net4=TemporalConvNet(TCN_NUM,TCN_HIDDEN_NUM)
    net4_face=TemporalConvNet(TCN_NUM,TCN_HIDDEN_NUM)
    net4_voice=TemporalConvNet(TCN_NUM,TCN_HIDDEN_NUM)

    net5 = Regressor(HIDDEN_NUM*2,HIDDEN_NUM*2)
    net5_face=Regressor(HIDDEN_NUM,HIDDEN_NUM)
    net5_voice=Regressor(HIDDEN_NUM,HIDDEN_NUM)

    nets=[net2_face,net2_voice,net3_face,net3_voice,net4,net4_face,net4_voice,net5,net5_face,net5_voice]
    for j in range(len(nets)):
        #net=nn.DataParallel(net)
        nets[j].load_state_dict(checkpoint['nets'][j])
        nets[j]=nets[j].cuda()
    w=checkpoint['w']
        
    test_datasets=[FVDataset(1,TRAIN_RIO,DATA_PATHS,tcn_num=TCN_NUM,person_test=1),FVDataset(0,TRAIN_RIO,DATA_PATHS,tcn_num=TCN_NUM,person_test=1)]

    test_dataloaders=[]
    for test_dataset in test_datasets:
        test_dataloaders.append(DataLoader(test_dataset, batch_size=1,shuffle=False))
    for test_dataloader in test_dataloaders:
        sum_losses = [0.0]*4
        cnt=0
        for net in nets:
            net.eval()
        with torch.no_grad():
            for i,data in enumerate(test_dataloader):
                xss,npys,y=data.values()
                y=float(y)
                losses = [[],[],[],[]]
                
                for j in range(len(xss)):
                    feas_face=[]
                    feas_voice=[]
                    for x in xss[j]:
                        x = x.cuda()
                        x = nets[0](x)
                        if VGG_OR_RESNET:
                            x=x.squeeze(-1).squeeze(-1)
                        _,fea = nets[2](x)
                        feas_face.append(fea)
                    for x in npys[j]:
                        x = x.cuda()
                        x = nets[1](x)
                        if VGG_OR_RESNET:
                            x=x.squeeze(-1).squeeze(-1)
                        _,fea = nets[3](x)
                        feas_voice.append(fea)

                    feas_face=torch.stack(feas_face)
                    feas_voice=torch.stack(feas_voice)
                    feas=torch.cat([feas_face,feas_voice],-1)
                    
                    feas_face = nets[5](feas_face).squeeze(1)
                    outputs_face,_=nets[8](feas_face)
                    losses[0].append(float(outputs_face.cpu()))

                    feas_voice = nets[6](feas_voice).squeeze(1)
                    outputs_voice,_=nets[9](feas_voice)
                    losses[1].append(float(outputs_voice.cpu()))

                    feas = nets[4](feas).squeeze(1)
                    outputs,_=nets[7](feas)
                    losses[2].append(float(outputs.cpu()))

                    losses[3].append(0.0)
                    for k in range(3):
                        losses[3][-1]+=losses[k][-1]*w[k]

                for k in range(4):
                    loss=losses[k]
                    loss.sort()
                    half=len(loss)//2
                    loss_mid=(loss[half]+loss[~half])/2
                    sum_losses[k]+=abs(loss_mid-y)

                cnt+=1
                    
        print('  [Person Test] Loss_face: %.03f Loss_voice: %.03f Loss_fv: %.03f Loss_fv_sum: %.03f'
                  % (sum_losses[0] / cnt, sum_losses[1] / cnt, sum_losses[2] / cnt, sum_losses[3] / cnt))

def vggTest():
    checkpoint = torch.load(os.path.join(LOGS_ROOT, MODEL_NAME))
    print(checkpoint["acc"])
    resnet=torchvision.models.resnet18(pretrained=True)
    if FACE_OR_VOICE:
        resnet.conv1= nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
    net2=nn.Sequential(*list(resnet.children())[:-1])

    net2.load_state_dict(checkpoint['net2'])
    net2 = net2.cuda()
    net3 = Regressor(EXTRACT_NUM,HIDDEN_NUM)
    net3.load_state_dict(checkpoint['net3'])
    net3 = net3.cuda() 

    if not FACE_OR_VOICE:
        test_dataset=FaceDataset(0,TRAIN_RIO,DATA_PATHS,is_person=0)
    else:
        test_dataset=VoiceDataset(0,TRAIN_RIO,DATA_PATHS,is_person=0)

    test_dataloader = DataLoader(test_dataset, batch_size=1,shuffle=False)

    sum_loss = 0.0

    cnt=0
    net2.eval()
    net3.eval()
    with torch.no_grad():
        for i,data in enumerate(test_dataloader):
            x,npy,y=data.values()
            y=y.to(torch.float32).unsqueeze(1)
            x, y = x.cuda(), y.cuda()

            
            x = net2(x)
            if VGG_OR_RESNET:
                x=x.squeeze(-1).squeeze(-1)

            outputs,_ = net3(x)
            
            #loss+=float(outputs.cpu())
            #loss=max(loss,float(outputs.cpu()))
            loss=float(outputs.cpu())
            # half=len(loss)//2
            # loss=(loss[half]+loss[~half])/2
            #loss/=len(xss)
            cnt+=1
            #print(y,loss)
            sum_loss+=abs(loss-y)
            # 每训练1个batch打印一次loss和准确率
            
    print('  [Test] Loss: %.03f'
            % (sum_loss / cnt))

def vggTrain():
    if not VGG_OR_RESNET:
        net2 = VGG("VGG19")
        checkpoint = torch.load(os.path.join(MODEL_ROOT, 'PrivateTest_model.t7'))
        net2.load_state_dict(checkpoint['net'])
        for para in net2.named_parameters():
            if para[0].startswith("features"):
                if int(para[0].split(".")[1])<26:
                    para[1].requires_grad = False
    else:
        resnet=torchvision.models.resnet18(pretrained=True)
        if FACE_OR_VOICE:
            resnet.conv1= nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
        net2=nn.Sequential(*list(resnet.children())[:-1])

    net2 = net2.cuda()
    net3 = Regressor(EXTRACT_NUM,HIDDEN_NUM).cuda() 

    # checkpoint = torch.load(os.path.join(LOGS_ROOT, CHECKPOINT_NAME[0]))
    # net2.load_state_dict(checkpoint['net2'])
    # net3.load_state_dict(checkpoint['net3'])
    
    
    criterion = nn.MSELoss() 
    
    if not FACE_OR_VOICE:
        train_dataset=FaceDataset(1,TRAIN_RIO,DATA_PATHS,is_person=0)
        test_dataset=FaceDataset(0,TRAIN_RIO,DATA_PATHS,is_person=0)
    else:
        train_dataset=VoiceDataset(1,TRAIN_RIO,DATA_PATHS,is_person=0)
        test_dataset=VoiceDataset(0,TRAIN_RIO,DATA_PATHS,is_person=0)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,shuffle=False)
    testloss_best=1e9

    for epoch in range(pre_epoch, EPOCH):

        optimizer = optim.Adam(chain(filter(lambda p: p.requires_grad, net2.parameters()),net3.parameters()), lr=LR,weight_decay=WEIGHT_DELAY)
        print('\nEpoch: %d' % (epoch + 1))
        
        sum_loss = 0.0

        net2.train()
        net3.train()
        for i,data in enumerate(train_dataloader):
            x,npy,y=data.values()
            y=y.to(torch.float32).unsqueeze(1)
            #print(y)
            y = y + 0.05*torch.randn(y.size()[0],1)

            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()

            
            x = net2(x)
            if VGG_OR_RESNET:
                x=x.squeeze(-1).squeeze(-1)

            outputs,_ = net3(x)
            
            loss = criterion(outputs, y)
          
            loss.backward()
            optimizer.step()

            # 每训练1个batch打印一次loss和准确率
            sum_loss += loss.item()
            #_, predicted = outputs.data
            #total += y.size(0)
            #correct += predicted.eq(y.data).cpu().sum()
            
            #sys.stdout.flush()
            if i>0:
                sys.stdout.write('\r')
            sys.stdout.write('[epoch:%d, iter:%d] Loss: %.03f'
                  % (epoch + 1, (i + 1 ), sum_loss / (i + 1)))
            
        
            # print('  loss:',fc_loss_all/(i+1),' acc:',acc/((i+1)*BATCH_SIZE))
        ####################################################################################
        sum_loss = 0.0

        cnt=0
        net2.eval()
        net3.eval()
        with torch.no_grad():
            for i,data in enumerate(test_dataloader):
                x,npy,y=data.values()
                y=y.to(torch.float32).unsqueeze(1)
                x, y = x.cuda(), y.cuda()

                
                x = net2(x)
                if VGG_OR_RESNET:
                    x=x.squeeze(-1).squeeze(-1)

                outputs,_ = net3(x)
                
                loss = criterion(outputs, y)
                cnt+=1
                # 每训练1个batch打印一次loss和准确率
                sum_loss += loss.item()
                #predicted = outputs.data
                # total += y.size(0)
                # correct += predicted.eq(y.data).cpu().sum()

        print('  [Test] Loss: %.03f'
                  % (sum_loss / cnt))

        testloss = sum_loss / cnt
        if testloss < testloss_best:
            testloss_best = testloss
            state = {
            'net2': net2.state_dict(),
            'net3': net3.state_dict(),
            'acc': testloss_best,
            'epoch': epoch,
        }
            torch.save(state, os.path.join(LOGS_ROOT,MODEL_NAME))            
    print(testloss_best)

def tcnTrain():
    if not VGG_OR_RESNET:
        net2 = VGG("VGG19")
        checkpoint = torch.load(os.path.join(LOGS_ROOT, CHECKPOINT_NAME[0]))
        
    else:
        resnet=torchvision.models.resnet18(pretrained=True)
        if FACE_OR_VOICE:
            resnet.conv1= nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
        net2=nn.Sequential(*list(resnet.children())[:-1])
        checkpoint = torch.load(os.path.join(LOGS_ROOT, CHECKPOINT_NAME[0]))

    

    net2.load_state_dict(checkpoint['net2'])
    net2 = net2.cuda()

    net3 = Regressor(EXTRACT_NUM,HIDDEN_NUM)
    net3.load_state_dict(checkpoint['net3'])
    net3 = net3.cuda()
    
    # if not TCN_OR_LSTM:
    net4=TemporalConvNet(TCN_NUM,TCN_HIDDEN_NUM)

    # else:
    #     net4=BILSTM(HIDDEN_NUM,HIDDEN_NUM,2,device)

    net4 = net4.cuda()

    net5 = Regressor(HIDDEN_NUM,HIDDEN_NUM)
    net5 = net5.cuda()

    # net4.load_state_dict(checkpoint['net4'])
    # net5.load_state_dict(checkpoint['net5'])
    # testloss_best=(checkpoint['acc'])

    criterion = nn.MSELoss() 
    
    if not FACE_OR_VOICE:
        train_dataset=FaceDataset(1,TRAIN_RIO,DATA_PATHS,is_person=1,tcn_num=TCN_NUM)
        test_dataset=FaceDataset(0,TRAIN_RIO,DATA_PATHS,is_person=1,tcn_num=TCN_NUM)
    else:
        train_dataset=VoiceDataset(1,TRAIN_RIO,DATA_PATHS,is_person=1,tcn_num=TCN_NUM)
        test_dataset=VoiceDataset(0,TRAIN_RIO,DATA_PATHS,is_person=1,tcn_num=TCN_NUM)
    
    train_dataloader = DataLoader(train_dataset, batch_size=TCN_BATCH_SIZE,shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=TCN_BATCH_SIZE,shuffle=False)
    testloss_best=1e9

    y1=[]
    y2=[]
    tmp=0.0

    for epoch in range(pre_epoch, EPOCH):

        optimizer = optim.Adam(chain(net4.parameters(),net5.parameters()), lr=LR,weight_decay=WEIGHT_DELAY)
        print('\nEpoch: %d' % (epoch + 1))
        
        sum_loss = 0.0

        net2.eval()
        net3.eval()
        net4.train()
        net5.train()

        for i,data in enumerate(train_dataloader):
            xs,npys,y=data.values()
            y=y.to(torch.float32).unsqueeze(1)
            y = y + 0.05*torch.randn(y.size()[0],1)
            y=y.cuda()
            feas=[]
            
            optimizer.zero_grad()
            for x in xs:
                with torch.no_grad():
                    x = x.cuda()
                    x = net2(x)
                    if VGG_OR_RESNET:
                        x=x.squeeze(-1).squeeze(-1)
                    _,fea = net3(x)
                    feas.append(fea)

            feas=torch.stack(feas)
            
            feas = net4(feas).squeeze(1)
            outputs,_=net5(feas)
            
            loss = criterion(outputs, y)
          
            loss.backward()
            optimizer.step()

            # 每训练1个batch打印一次loss和准确率
            sum_loss += loss.item()

            tmp=sum_loss / (i + 1)
            if i>0:
                sys.stdout.write('\r')
            sys.stdout.write('[epoch:%d, iter:%d] Loss: %.03f'
                  % (epoch + 1, (i + 1 ),tmp))
            
        y1.append(tmp)
            # print('  loss:',fc_loss_all/(i+1),' acc:',acc/((i+1)*BATCH_SIZE))
        ####################################################################################
        sum_loss = 0.0

        cnt=0
        net2.eval()
        net3.eval()
        net4.eval()
        net5.eval()
        with torch.no_grad():
            for i,data in enumerate(test_dataloader):
                xs,npys,y=data.values()
                y=y.to(torch.float32).unsqueeze(1)
                y=y.cuda()
                feas=[]
                
                for x in xs:
                    x = x.cuda()
                    x = net2(x)
                    if VGG_OR_RESNET:
                        x=x.squeeze(-1).squeeze(-1)
                    _,fea = net3(x)
                    feas.append(fea)

                feas=torch.stack(feas)
                feas = net4(feas).squeeze(1)
                outputs,_=net5(feas)
                
                loss = criterion(outputs, y)
                cnt+=1
                # 每训练1个batch打印一次loss和准确率
                sum_loss += loss.item()
                #predicted = outputs.data
                # total += y.size(0)
                # correct += predicted.eq(y.data).cpu().sum()
        
        testloss = sum_loss / cnt
        print('  [Test] Loss: %.03f'
                  % (testloss))
        y2.append(testloss)

        if testloss < testloss_best:
            testloss_best = testloss
            state = {
            'net2': net2.state_dict(),
            'net3': net3.state_dict(),
            'net4': net4.state_dict(),
            'net5': net5.state_dict(),
            'acc': testloss_best,
            'epoch': epoch,
        }
            torch.save(state, os.path.join(LOGS_ROOT,MODEL_NAME))        

    drawplt(y1,y2)    
    print(testloss_best)

def calculate_W(pre_epoch,EPOCH,checkpoint,K,train_dataloader,test_dataloader):
    resnet_face=torchvision.models.resnet18(pretrained=True)
    net2_face=nn.Sequential(*list(resnet_face.children())[:-1])

    resnet_voice=torchvision.models.resnet18(pretrained=True)
    resnet_voice.conv1= nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
    net2_voice=nn.Sequential(*list(resnet_voice.children())[:-1])

    net3_face = Regressor(EXTRACT_NUM,HIDDEN_NUM)
    
    net3_voice = Regressor(EXTRACT_NUM,HIDDEN_NUM)


    # resnet_face_fv=torchvision.models.resnet18(pretrained=True)
    # net2_face_fv=nn.Sequential(*list(resnet_face_fv.children())[:-1])

    # resnet_voice_fv=torchvision.models.resnet18(pretrained=True)
    # resnet_voice_fv.conv1= nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
    # net2_voice_fv=nn.Sequential(*list(resnet_voice_fv.children())[:-1])
    
    # net3_face_fv = Regressor(EXTRACT_NUM,HIDDEN_NUM)
    
    # net3_voice_fv = Regressor(EXTRACT_NUM,HIDDEN_NUM)


    net4=TemporalConvNet(TCN_NUM,TCN_HIDDEN_NUM)
    net4_face=TemporalConvNet(TCN_NUM,TCN_HIDDEN_NUM)
    net4_voice=TemporalConvNet(TCN_NUM,TCN_HIDDEN_NUM)

    net5 = Regressor(HIDDEN_NUM*2,HIDDEN_NUM*2)
    net5_face=Regressor(HIDDEN_NUM,HIDDEN_NUM)
    net5_voice=Regressor(HIDDEN_NUM,HIDDEN_NUM)

    nets=[net2_face,net2_voice,net3_face,net3_voice,net4,net4_face,net4_voice,net5,net5_face,net5_voice]
    for j in range(len(nets)):
        nets[j].load_state_dict(checkpoint['nets'][j])
        #nets[j]=nn.DataParallel(nets[j])
        nets[j]=nets[j].cuda()
    # for j in range(10,14):
    #     nets[j].load_state_dict(checkpoint['nets'][j-10])
    #     #nets[j]=nn.DataParallel(nets[j])
    #     nets[j]=nets[j].cuda()


    criterion = nn.MSELoss() 
    optimizer_face = optim.Adam(chain(*[net.parameters() for net in [nets[5],nets[8]]]), lr=LR,weight_decay=WEIGHT_DELAY)
    optimizer_voice = optim.Adam(chain(*[net.parameters() for net in [nets[6],nets[9]]]), lr=LR,weight_decay=WEIGHT_DELAY)
    optimizer_fv = optim.Adam(chain(*[net.parameters() for net in [nets[4],nets[7]]]), lr=LR,weight_decay=WEIGHT_DELAY)
    optimizers=[optimizer_face,optimizer_voice,optimizer_fv]
    loss_history=[[],[],[]]
    
    for epoch in range(pre_epoch, EPOCH):
        
        sum_losses=[0.0]*3
        losses=[0.0]*3
        cnt=0

        for net in nets[:4]:
            net.eval()
        for net in nets[4:]:
            net.train()

        for i,data in enumerate(train_dataloader):
            xs,npys,y=data.values()
            y=y.to(torch.float32).unsqueeze(1)
            y = y + 0.05*torch.randn(y.size()[0],1)
            y=y.cuda()
            feas_face=[]
            feas_voice=[]
            # xs2=xs.clone()
            # npys2=npys.clone()
            
            for x in xs:
                with torch.no_grad():
                    x = x.cuda()
                    x = nets[0](x)
                    if VGG_OR_RESNET:
                        x=x.squeeze(-1).squeeze(-1)
                    _,fea = nets[2](x)
                    feas_face.append(fea)

            for x in npys:
                with torch.no_grad():
                    x = x.cuda()
                    x = nets[1](x)
                    if VGG_OR_RESNET:
                        x=x.squeeze(-1).squeeze(-1)
                    _,fea = nets[3](x)
                    feas_voice.append(fea)

            feas_face=torch.stack(feas_face)
            feas_voice=torch.stack(feas_voice)
            feas=torch.cat([feas_face,feas_voice],-1)

            feas_face = nets[5](feas_face).squeeze(1)
            outputs_face,_=nets[8](feas_face)
            losses[0] = criterion(outputs_face, y)

            feas_voice = nets[6](feas_voice).squeeze(1)
            outputs_voice,_=nets[9](feas_voice)
            losses[1] = criterion(outputs_voice, y)

            # feas_face=[]
            # feas_voice=[]
            
            # for x in xs2:
            #     # with torch.no_grad():
            #     x = x.cuda()
            #     x = nets[10](x)
            #     if VGG_OR_RESNET:
            #         x=x.squeeze(-1).squeeze(-1)
            #     _,fea = nets[12](x)
            #     feas_face.append(fea)

            # for x in npys2:
            #     # with torch.no_grad():
            #     x = x.cuda()
            #     x = nets[11](x)
            #     if VGG_OR_RESNET:
            #         x=x.squeeze(-1).squeeze(-1)
            #     _,fea = nets[13](x)
            #     feas_voice.append(fea)

            # feas_face=torch.stack(feas_face)
            # feas_voice=torch.stack(feas_voice)
            # feas=torch.cat([feas_face,feas_voice],-1)

            feas = nets[4](feas).squeeze(1)
            outputs,_=nets[7](feas)
            losses[2] = criterion(outputs, y)

            for j in range(K):
                optimizers[j].zero_grad()
                losses[j].backward()
                optimizers[j].step()
                sum_losses[j] += losses[j].item()
            
            cnt+=1

            if i>0:
                sys.stdout.write('\r')
            sys.stdout.write('calW(%d,%d)[epoch:%d, iter:%d] Loss_face: %.03f Loss_voice: %.03f Loss_fv: %.03f'
                  % (pre_epoch,EPOCH,epoch + 1, (i + 1 ), sum_losses[0] / (i + 1),sum_losses[1] / (i + 1),sum_losses[2] / (i + 1)))
        
        for j in range(K):
            sum_losses[j]/=cnt

        sum_losses_test=[0.0]*3

        cnt=0
        for net in nets:
            net.eval()
        with torch.no_grad():
            for i,data in enumerate(test_dataloader):
                xs,npys,y=data.values()
                y=y.to(torch.float32).unsqueeze(1)
                y=y.cuda()
                feas_face=[]
                feas_voice=[]
                xs2=xs.clone()
                npys2=npys.clone()

                for x in xs:
                    x = x.cuda()
                    x = nets[0](x)
                    if VGG_OR_RESNET:
                        x=x.squeeze(-1).squeeze(-1)
                    _,fea = nets[2](x)
                    feas_face.append(fea)

                for x in npys:
                    x = x.cuda()
                    x = nets[1](x)
                    if VGG_OR_RESNET:
                        x=x.squeeze(-1).squeeze(-1)
                    _,fea = nets[3](x)
                    feas_voice.append(fea)

                feas_face=torch.stack(feas_face)
                feas_voice=torch.stack(feas_voice)
                feas=torch.cat([feas_face,feas_voice],-1)

                feas_face = nets[5](feas_face).squeeze(1)
                outputs_face,_=nets[8](feas_face)
                losses[0] = criterion(outputs_face, y)

                feas_voice = nets[6](feas_voice).squeeze(1)
                outputs_voice,_=nets[9](feas_voice)
                losses[1] = criterion(outputs_voice, y)


                # feas_face=[]
                # feas_voice=[]
                
                # for x in xs2:
                #     # with torch.no_grad():
                #     x = x.cuda()
                #     x = nets[10](x)
                #     if VGG_OR_RESNET:
                #         x=x.squeeze(-1).squeeze(-1)
                #     _,fea = nets[12](x)
                #     feas_face.append(fea)

                # for x in npys2:
                #     # with torch.no_grad():
                #     x = x.cuda()
                #     x = nets[11](x)
                #     if VGG_OR_RESNET:
                #         x=x.squeeze(-1).squeeze(-1)
                #     _,fea = nets[13](x)
                #     feas_voice.append(fea)

                # feas_face=torch.stack(feas_face)
                # feas_voice=torch.stack(feas_voice)
                # feas=torch.cat([feas_face,feas_voice],-1)

                feas = nets[4](feas).squeeze(1)
                outputs,_=nets[7](feas)
                losses[2] = criterion(outputs, y)

                cnt+=1

                for j in range(K):
                    sum_losses_test[j] += losses[j].item()

        print('  [Test] Loss_face: %.03f Loss_voice: %.03f Loss_fv: %.03f'
                  % (sum_losses_test[0] / cnt, sum_losses_test[1] / cnt, sum_losses_test[2] / cnt))

        for j in range(K):
            sum_losses_test[j]/=cnt          
        for j in range(K):
            loss_history[j].append([sum_losses[j],sum_losses_test[j]])

    O=[0.0]*3
    G=[0.0]*3
    w=[0.0]*3
    
    inf=1e-9
    Z=inf

    for j in range(K):
        O[j]=(loss_history[j][-1][-1]-loss_history[j][-1][0])-(loss_history[j][0][-1]-loss_history[j][0][0])+inf
        G[j]=loss_history[j][-1][-1]-loss_history[j][0][-1]+inf
        O[j]=abs(O[j])
        G[j]=abs(G[j])
        Z+=G[j]/(O[j]*O[j])

    for j in range(K):
        w[j]=(1.0/Z)*(G[j]/(O[j]*O[j]))
    print("w: ",w)
    return w

def fvTrain():
    checkpoint = torch.load(os.path.join(LOGS_ROOT, "v8_fv.t7"))
    # checkpoint_voice = torch.load(os.path.join(LOGS_ROOT, "v4_voice_res.t7"))
    resnet_face=torchvision.models.resnet18(pretrained=True)
    net2_face=nn.Sequential(*list(resnet_face.children())[:-1])

    resnet_voice=torchvision.models.resnet18(pretrained=True)
    resnet_voice.conv1= nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
    net2_voice=nn.Sequential(*list(resnet_voice.children())[:-1])

    net3_face = Regressor(EXTRACT_NUM,HIDDEN_NUM)
    
    net3_voice = Regressor(EXTRACT_NUM,HIDDEN_NUM)

    # net2_face.load_state_dict(checkpoint_face["net2"])
    # net3_face.load_state_dict(checkpoint_face["net3"])
    # net2_voice.load_state_dict(checkpoint_voice["net2"])
    # net3_voice.load_state_dict(checkpoint_voice["net3"])

    net4=TemporalConvNet(TCN_NUM,TCN_HIDDEN_NUM)
    net4_face=TemporalConvNet(TCN_NUM,TCN_HIDDEN_NUM)
    net4_voice=TemporalConvNet(TCN_NUM,TCN_HIDDEN_NUM)

    net5 = Regressor(HIDDEN_NUM*2,HIDDEN_NUM*2)
    net5_face=Regressor(HIDDEN_NUM,HIDDEN_NUM)
    net5_voice=Regressor(HIDDEN_NUM,HIDDEN_NUM)

    nets=[net2_face,net2_voice,net3_face,net3_voice,net4,net4_face,net4_voice,net5,net5_face,net5_voice]
    # for net in nets:
    #     #net=nn.DataParallel(net)
    #     net=net.cuda()

    for j in range(len(nets)):
        nets[j].load_state_dict(checkpoint['nets'][j])
        nets[j]=nets[j].cuda()

    criterion = nn.MSELoss() 
    optimizer = optim.Adam(chain(*[net.parameters() for net in nets[4:]]), lr=LR,weight_decay=WEIGHT_DELAY)
    
    train_dataset=FVDataset(1,TRAIN_RIO,DATA_PATHS,tcn_num=TCN_NUM)
    test_dataset=FVDataset(0,TRAIN_RIO,DATA_PATHS,tcn_num=TCN_NUM)

    train_dataloader = DataLoader(train_dataset, batch_size=TCN_BATCH_SIZE,shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=TCN_BATCH_SIZE,shuffle=False)
    testloss_best=1e9
    testloss_best=checkpoint['acc']
    w=[0.7,0.1,0.2]

    for epoch in range(pre_epoch, EPOCH):

        # if epoch%SUB_EPOCH==0:
        #     state = {
        #     'nets': [net.state_dict() for net in nets],
        #     'epoch': epoch,
        # }
        #     torch.save(state, os.path.join(LOGS_ROOT,"init_"+MODEL_NAME))   
        #     checkpoint = torch.load(os.path.join(LOGS_ROOT, "init_"+MODEL_NAME))
        #     w=calculate_W(epoch,epoch+SUB_EPOCH,checkpoint,3,train_dataloader,test_dataloader)
        
        sum_loss = 0.0

        for net in nets[:4]:
            net.eval()
        for net in nets[4:]:
            net.train()

        for i,data in enumerate(train_dataloader):
            xs,npys,y=data.values()
            y=y.to(torch.float32).unsqueeze(1)
            y = y + 0.05*torch.randn(y.size()[0],1)
            y=y.cuda()
            feas_face=[]
            feas_voice=[]
            
            optimizer.zero_grad()
            for x in xs:
                with torch.no_grad():
                    x = x.cuda()
                    x = nets[0](x)
                    if VGG_OR_RESNET:
                        x=x.squeeze(-1).squeeze(-1)
                    _,fea = nets[2](x)
                    feas_face.append(fea)

            for x in npys:
                with torch.no_grad():
                    x = x.cuda()
                    x = nets[1](x)
                    if VGG_OR_RESNET:
                        x=x.squeeze(-1).squeeze(-1)
                    _,fea = nets[3](x)
                    feas_voice.append(fea)

            feas_face=torch.stack(feas_face)
            feas_voice=torch.stack(feas_voice)
            feas=torch.cat([feas_face,feas_voice],-1)

            feas_face = nets[5](feas_face).squeeze(1)
            outputs_face,_=nets[8](feas_face)
            losses0 = criterion(outputs_face, y)

            feas_voice = nets[6](feas_voice).squeeze(1)
            outputs_voice,_=nets[9](feas_voice)
            losses1 = criterion(outputs_voice, y)

            feas = nets[4](feas).squeeze(1)
            outputs,_=nets[7](feas)
            losses2 = criterion(outputs, y)
            
            loss = losses0*w[0]+losses1*w[1]+losses2*w[2]
          
            loss.backward()
            optimizer.step()

            # 每训练1个batch打印一次loss和准确率
            sum_loss += loss.item()
            #_, predicted = outputs.data
            #total += y.size(0)
            #correct += predicted.eq(y.data).cpu().sum()
            
            #sys.stdout.flush()
            if i>0:
                sys.stdout.write('\r')
            sys.stdout.write('main[epoch:%d, iter:%d] Loss: %.03f'
                  % (epoch + 1, (i + 1 ), sum_loss / (i + 1)))
            
        
            # print('  loss:',fc_loss_all/(i+1),' acc:',acc/((i+1)*BATCH_SIZE))
        ####################################################################################
        sum_loss = 0.0

        cnt=0
        for net in nets:
            net.eval()
        with torch.no_grad():
            for i,data in enumerate(test_dataloader):
                xs,npys,y=data.values()
                y=y.to(torch.float32).unsqueeze(1)
                y=y.cuda()
                feas_face=[]
                feas_voice=[]
                
                for x in xs:
                    x = x.cuda()
                    x = nets[0](x)
                    if VGG_OR_RESNET:
                        x=x.squeeze(-1).squeeze(-1)
                    _,fea = nets[2](x)
                    feas_face.append(fea)

                for x in npys:
                    x = x.cuda()
                    x = nets[1](x)
                    if VGG_OR_RESNET:
                        x=x.squeeze(-1).squeeze(-1)
                    _,fea = nets[3](x)
                    feas_voice.append(fea)

                feas_face=torch.stack(feas_face)
                feas_voice=torch.stack(feas_voice)
                feas=torch.cat([feas_face,feas_voice],-1)

                feas_face = nets[5](feas_face).squeeze(1)
                outputs_face,_=nets[8](feas_face)
                losses0 = criterion(outputs_face, y)

                feas_voice = nets[6](feas_voice).squeeze(1)
                outputs_voice,_=nets[9](feas_voice)
                losses1 = criterion(outputs_voice, y)

                feas = nets[4](feas).squeeze(1)
                outputs,_=nets[7](feas)
                losses2 = criterion(outputs, y)
                
                loss = losses0*w[0]+losses1*w[1]+losses2*w[2]

                cnt+=1
                # 每训练1个batch打印一次loss和准确率
                sum_loss += loss.item()
                #predicted = outputs.data
                # total += y.size(0)
                # correct += predicted.eq(y.data).cpu().sum()

        print('  [Test] Loss: %.03f'
                  % (sum_loss / cnt))

        testloss = sum_loss / cnt
        if testloss < testloss_best:
            testloss_best = testloss
            state = {
            'nets': [net.state_dict() for net in nets],
            'acc': testloss_best,
            'epoch': epoch,
            'w':w
        }
            torch.save(state, os.path.join(LOGS_ROOT,MODEL_NAME))            
    print(testloss_best,w)

def fvTrain_origin():
    checkpoint1 = torch.load(os.path.join(LOGS_ROOT, 'v7_face.t7'))

    checkpoint2 = torch.load(os.path.join(LOGS_ROOT, 'v7_voice.t7'))


    resnet_face=torchvision.models.resnet18(pretrained=True)
    net2_face=nn.Sequential(*list(resnet_face.children())[:-1])
    net2_face.load_state_dict(checkpoint1['net2'])
    net2_face = net2_face.cuda()

    resnet_voice=torchvision.models.resnet18(pretrained=True)
    resnet_voice.conv1= nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
    net2_voice=nn.Sequential(*list(resnet_voice.children())[:-1])
    net2_voice.load_state_dict(checkpoint2['net2'])
    net2_voice = net2_voice.cuda()

    net3_face = Regressor(EXTRACT_NUM,HIDDEN_NUM)
    net3_face.load_state_dict(checkpoint1['net3'])
    net3_face = net3_face.cuda()
    
    net3_voice = Regressor(EXTRACT_NUM,HIDDEN_NUM)
    net3_voice.load_state_dict(checkpoint2['net3'])
    net3_voice = net3_voice.cuda()

    net4=TemporalConvNet(TCN_NUM,TCN_HIDDEN_NUM)

    net4 = net4.cuda()

    net5 = Regressor(HIDDEN_NUM*2,HIDDEN_NUM*2)
    net5 = net5.cuda()


    criterion = nn.MSELoss() 
    
    train_dataset=FVDataset(1,TRAIN_RIO,DATA_PATHS,tcn_num=TCN_NUM)
    test_dataset=FVDataset(0,TRAIN_RIO,DATA_PATHS,tcn_num=TCN_NUM)

    train_dataloader = DataLoader(train_dataset, batch_size=TCN_BATCH_SIZE,shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=TCN_BATCH_SIZE,shuffle=False)
    testloss_best=1e9

    for epoch in range(pre_epoch, EPOCH):

        optimizer = optim.Adam(chain(net4.parameters(),net5.parameters()), lr=LR,weight_decay=WEIGHT_DELAY)
        print('\nEpoch: %d' % (epoch + 1))
        
        sum_loss = 0.0

        net2_face.eval()
        net3_face.eval()
        net2_voice.eval()
        net3_voice.eval()
        net4.train()
        net5.train()

        for i,data in enumerate(train_dataloader):
            xs,npys,y=data.values()
            y=y.to(torch.float32).unsqueeze(1)
            y = y + 0.05*torch.randn(y.size()[0],1)
            y=y.cuda()
            feas_face=[]
            feas_voice=[]
            
            optimizer.zero_grad()
            for x in xs:
                with torch.no_grad():
                    x = x.cuda()
                    x = net2_face(x)
                    if VGG_OR_RESNET:
                        x=x.squeeze(-1).squeeze(-1)
                    _,fea = net3_face(x)
                    feas_face.append(fea)

            for x in npys:
                with torch.no_grad():
                    x = x.cuda()
                    x = net2_voice(x)
                    if VGG_OR_RESNET:
                        x=x.squeeze(-1).squeeze(-1)
                    _,fea = net3_voice(x)
                    feas_voice.append(fea)

            feas_face=torch.stack(feas_face)
            feas_voice=torch.stack(feas_voice)
            feas=torch.cat([feas_face,feas_voice],-1)

            feas = net4(feas).squeeze(1)
            outputs,_=net5(feas)
            
            loss = criterion(outputs, y)
          
            loss.backward()
            optimizer.step()

            # 每训练1个batch打印一次loss和准确率
            sum_loss += loss.item()
            #_, predicted = outputs.data
            #total += y.size(0)
            #correct += predicted.eq(y.data).cpu().sum()
            
            #sys.stdout.flush()
            if i>0:
                sys.stdout.write('\r')
            sys.stdout.write('[epoch:%d, iter:%d] Loss: %.03f'
                  % (epoch + 1, (i + 1 ), sum_loss / (i + 1)))
            
        
            # print('  loss:',fc_loss_all/(i+1),' acc:',acc/((i+1)*BATCH_SIZE))
        ####################################################################################
        sum_loss = 0.0

        cnt=0
        net2_face.eval()
        net3_face.eval()
        net2_voice.eval()
        net3_voice.eval()
        net4.eval()
        net5.eval()
        with torch.no_grad():
            for i,data in enumerate(test_dataloader):
                xs,npys,y=data.values()
                y=y.to(torch.float32).unsqueeze(1)
                y=y.cuda()
                feas_face=[]
                feas_voice=[]
                
                for x in xs:
                    x = x.cuda()
                    x = net2_face(x)
                    if VGG_OR_RESNET:
                        x=x.squeeze(-1).squeeze(-1)
                    _,fea = net3_face(x)
                    feas_face.append(fea)

                for x in npys:
                    x = x.cuda()
                    x = net2_voice(x)
                    if VGG_OR_RESNET:
                        x=x.squeeze(-1).squeeze(-1)
                    _,fea = net3_voice(x)
                    feas_voice.append(fea)

                feas_face=torch.stack(feas_face)
                feas_voice=torch.stack(feas_voice)
                feas=torch.cat([feas_face,feas_voice],-1)

                feas = net4(feas).squeeze(1)
                outputs,_=net5(feas)
                
                loss = criterion(outputs, y)
                cnt+=1
                # 每训练1个batch打印一次loss和准确率
                sum_loss += loss.item()
                #predicted = outputs.data
                # total += y.size(0)
                # correct += predicted.eq(y.data).cpu().sum()

        print('  [Test] Loss: %.03f'
                  % (sum_loss / cnt))

        testloss = sum_loss / cnt
        if testloss < testloss_best:
            testloss_best = testloss
            state = {
            'net2_face': net2_face.state_dict(),
            'net3_face': net3_face.state_dict(),
            'net2_voice': net2_voice.state_dict(),
            'net3_voice': net3_voice.state_dict(),
            'net4': net4.state_dict(),
            'net5': net5.state_dict(),
            'acc': testloss_best,
            'epoch': epoch,
        }
            torch.save(state, os.path.join(LOGS_ROOT,MODEL_NAME))            
    print(testloss_best)

def fvTest_detail():
    checkpoint_face = torch.load(os.path.join(LOGS_ROOT, CHECKPOINT_NAME[0]))
    checkpoint_voice = torch.load(os.path.join(LOGS_ROOT, CHECKPOINT_NAME[1]))
    print(checkpoint_face["acc"])
    print(checkpoint_voice["acc"])

    resnet_face=torchvision.models.resnet18(pretrained=True)
    net2_face=nn.Sequential(*list(resnet_face.children())[:-1])
    net2_face.load_state_dict(checkpoint_face["net2"])

    resnet_voice=torchvision.models.resnet18(pretrained=True)
    resnet_voice.conv1= nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
    net2_voice=nn.Sequential(*list(resnet_voice.children())[:-1])
    net2_voice.load_state_dict(checkpoint_voice["net2"])

    net3_face = Regressor(EXTRACT_NUM,HIDDEN_NUM)
    net3_face.load_state_dict(checkpoint_face["net3"])
    
    net3_voice = Regressor(EXTRACT_NUM,HIDDEN_NUM)
    net3_voice.load_state_dict(checkpoint_voice["net3"])

    net4_face=TemporalConvNet(TCN_NUM,TCN_HIDDEN_NUM)
    net4_face.load_state_dict(checkpoint_face["net4"])
    
    net4_voice=TemporalConvNet(TCN_NUM,TCN_HIDDEN_NUM)
    net4_voice.load_state_dict(checkpoint_voice["net4"])

    net5_face=Regressor(HIDDEN_NUM,HIDDEN_NUM)
    net5_face.load_state_dict(checkpoint_face["net5"])
    
    net5_voice=Regressor(HIDDEN_NUM,HIDDEN_NUM)
    net5_voice.load_state_dict(checkpoint_voice["net5"])

    nets=[net2_face,net2_voice,net3_face,net3_voice,net4_face,net4_voice,net5_face,net5_voice]
    for j in range(len(nets)):
        #net=nn.DataParallel(net)
        nets[j]=nets[j].cuda()
        
    test_datasets=[FVDataset(0,TRAIN_RIO,DATA_PATHS,tcn_num=TCN_NUM,person_test=1)]

    test_dataloaders=[]
    for test_dataset in test_datasets:
        test_dataloaders.append(DataLoader(test_dataset, batch_size=1,shuffle=False))
    for test_dataloader in test_dataloaders:
        sum_loss = 0.0
        loss=0.0
        
        cnt=0
        for net in nets:
            net.eval()
        with torch.no_grad():
            for i,data in enumerate(test_dataloader):
                xss,npys,y=data.values()
                y=float(y)
                losses=[]
                for j in range(len(xss)):
                    feas_face=[]
                    feas_voice=[]
                    for x in xss[j]:
                        x = x.cuda()
                        x = nets[0](x)
                        if VGG_OR_RESNET:
                            x=x.squeeze(-1).squeeze(-1)
                        _,fea = nets[2](x)
                        feas_face.append(fea)
                    for x in npys[j]:
                        x = x.cuda()
                        x = nets[1](x)
                        if VGG_OR_RESNET:
                            x=x.squeeze(-1).squeeze(-1)
                        _,fea = nets[3](x)
                        feas_voice.append(fea)

                    feas_face=torch.stack(feas_face)
                    feas_voice=torch.stack(feas_voice)
                    
                    feas_face = nets[4](feas_face).squeeze(1)
                    outputs_face,_=nets[6](feas_face)
                    loss=(float(outputs_face.cpu())+0.5)/5.0


                    feas_voice = nets[5](feas_voice).squeeze(1)
                    outputs_voice,_=nets[7](feas_voice)
                    loss+=(float(outputs_voice.cpu())+0.5)/5.0


                    losses.append(loss)

                losses.sort()
                half=len(losses)//2
                loss_mid=(losses[half]+losses[~half])/2
                sum_loss+=abs(loss_mid*2.5-y)


                cnt+=1
                    
        print('  [Person Test] Loss_fv_sum: %.03f'
                  % (sum_loss / cnt))

def drawplt(y1,y2):
    x=range(0,EPOCH)
    plt.plot(x, y1,color='green', label='train')    
    plt.plot(x, y2, color='red', label='test')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.savefig(os.path.join(LOGS_ROOT,'loss.png'))

tcnTrain()