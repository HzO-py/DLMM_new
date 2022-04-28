# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 08:18:26 2020

@author: 98669
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import argparse
import numpy as np
import itertools
from tqdm import tqdm
import os
import sys
from torch.utils.data import DataLoader
from utils import getCfg
from models import BILSTM, Prototype,Classifier,ResNet18,cnn1d,VGG,Regressor,TemporalConvNet
from loader import BioDataset,FaceDataset,VoiceDataset,FVDataset
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch Biovid PreTraining')
parser.add_argument('--yamlFile', default='config/face_config.yaml', help='yaml file') 
args = parser.parse_args()

cfg=getCfg(args.yamlFile)

os.environ["CUDA_VISIBLE_DEVICES"] = cfg["GPUID"]
EPOCH = cfg["EPOCH"]
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



def test():
    checkpoint = torch.load(os.path.join(LOGS_ROOT, MODEL_NAME))
    print(checkpoint["acc"])
    resnet=torchvision.models.resnet18(pretrained=True)
    if FACE_OR_VOICE:
        resnet.conv1= nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
    net2=nn.Sequential(*list(resnet.children())[:-1])

    net2.load_state_dict(checkpoint['net2'])
    net2 = net2.to(device)
    net3 = Regressor(EXTRACT_NUM,HIDDEN_NUM)
    net3.load_state_dict(checkpoint['net3'])
    net3 = net3.to(device) 
    net4=TemporalConvNet(TCN_NUM,TCN_HIDDEN_NUM)
    net4.load_state_dict(checkpoint['net4'])
    net4 = net4.to(device)
    net5 = Regressor(HIDDEN_NUM,HIDDEN_NUM)
    net5.load_state_dict(checkpoint['net5'])
    net5 = net5.to(device)

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
        with torch.no_grad():
            for i,data in enumerate(test_dataloader):
                xss,npys,y=data.values()
                y=float(y)
                loss=[]
                
                for xs in xss:
                    feas=[]
                    for x in xs:
                        x = x.to(device)
                        x = net2(x)
                        x=x.squeeze(-1).squeeze(-1)
                        
                        _,fea = net3(x)
                        feas.append(fea)

                    feas=torch.stack(feas)
                    
                    feas = net4(feas).squeeze(1)
                    outputs,_=net5(feas)
                    #loss+=float(outputs.cpu())
                    #loss=max(loss,float(outputs.cpu()))
                    loss.append(float(outputs.cpu()))
                half=len(loss)//2
                loss=(loss[half]+loss[~half])/2
                #loss/=len(xss)
                cnt+=1
                #print(y,loss)
                sum_loss+=abs(loss-y)
                    # 每训练1个batch打印一次loss和准确率
                    
        print('  [Test] Loss: %.03f'
                  % (sum_loss / cnt))

def fvtest():
    checkpoint = torch.load(os.path.join(LOGS_ROOT, MODEL_NAME))
    print(checkpoint["acc"])
    
    resnet_face=torchvision.models.resnet18(pretrained=True)
    net2_face=nn.Sequential(*list(resnet_face.children())[:-1])
    net2_face.load_state_dict(checkpoint['net2_face'])
    net2_face = net2_face.to(device)

    resnet_voice=torchvision.models.resnet18(pretrained=True)
    resnet_voice.conv1= nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
    net2_voice=nn.Sequential(*list(resnet_voice.children())[:-1])
    net2_voice.load_state_dict(checkpoint['net2_voice'])
    net2_voice = net2_voice.to(device)

    net3_face = Regressor(EXTRACT_NUM,HIDDEN_NUM)
    net3_face.load_state_dict(checkpoint['net3_face'])
    net3_face = net3_face.to(device)
    
    net3_voice = Regressor(EXTRACT_NUM,HIDDEN_NUM)
    net3_voice.load_state_dict(checkpoint['net3_voice'])
    net3_voice = net3_voice.to(device)

    
    net4=TemporalConvNet(TCN_NUM,TCN_HIDDEN_NUM)
    net4.load_state_dict(checkpoint['net4'])
    net4 = net4.to(device)
    net5 = Regressor(HIDDEN_NUM*2,HIDDEN_NUM*2)
    net5.load_state_dict(checkpoint['net5'])
    net5 = net5.to(device)

    test_datasets=[FVDataset(0,TRAIN_RIO,DATA_PATHS,tcn_num=TCN_NUM,person_test=1)]

    test_dataloaders=[]
    for test_dataset in test_datasets:
        test_dataloaders.append(DataLoader(test_dataset, batch_size=1,shuffle=False))
    for test_dataloader in test_dataloaders:
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
                xss,npys,y=data.values()
                y=float(y)
                loss=[]
                
                for j in range(len(xss)):
                    feas_face=[]
                    feas_voice=[]
                    for x in xss[j]:
                        x = x.to(device)
                        x = net2_face(x)
                        if VGG_OR_RESNET:
                            x=x.squeeze(-1).squeeze(-1)
                        _,fea = net3_face(x)
                        feas_face.append(fea)
                    for x in npys[j]:
                        x = x.to(device)
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

                    #loss+=float(outputs.cpu())
                    #loss=max(loss,float(outputs.cpu()))
                    loss.append(float(outputs.cpu()))
                half=len(loss)//2
                loss=(loss[half]+loss[~half])/2
                #loss/=len(xss)
                cnt+=1
                #print(y,loss)
                sum_loss+=abs(loss-y)
                    # 每训练1个batch打印一次loss和准确率
                    
        print('  [Test] Loss: %.03f'
                  % (sum_loss / cnt))

def vggTest():
    checkpoint = torch.load(os.path.join(LOGS_ROOT, MODEL_NAME))
    print(checkpoint["acc"])
    resnet=torchvision.models.resnet18(pretrained=True)
    if FACE_OR_VOICE:
        resnet.conv1= nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
    net2=nn.Sequential(*list(resnet.children())[:-1])

    net2.load_state_dict(checkpoint['net2'])
    net2 = net2.to(device)
    net3 = Regressor(EXTRACT_NUM,HIDDEN_NUM)
    net3.load_state_dict(checkpoint['net3'])
    net3 = net3.to(device) 

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
            x, y = x.to(device), y.to(device)

            
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

    net2 = net2.to(device)
    net3 = Regressor(EXTRACT_NUM,HIDDEN_NUM).to(device) 
    
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

        optimizer = optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, net2.parameters()),net3.parameters()), lr=LR,weight_decay=WEIGHT_DELAY)
        print('\nEpoch: %d' % (epoch + 1))
        
        sum_loss = 0.0

        net2.train()
        net3.train()
        for i,data in enumerate(train_dataloader):
            x,npy,y=data.values()
            y=y.to(torch.float32).unsqueeze(1)
            y = y + 0.05*torch.randn(y.size()[0],1)

            x, y = x.to(device), y.to(device)
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
                x, y = x.to(device), y.to(device)

                
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
        checkpoint = torch.load(os.path.join(LOGS_ROOT, 'v4_face_vgg.t7'))
        
    else:
        resnet=torchvision.models.resnet18(pretrained=True)
        if FACE_OR_VOICE:
            resnet.conv1= nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
        net2=nn.Sequential(*list(resnet.children())[:-1])
        checkpoint = torch.load(os.path.join(LOGS_ROOT, 'v4_face_rgb_res.t7'))

    

    net2.load_state_dict(checkpoint['net2'])
    net2 = net2.to(device)

    net3 = Regressor(EXTRACT_NUM,HIDDEN_NUM)
    net3.load_state_dict(checkpoint['net3'])
    net3 = net3.to(device)
    
    if not TCN_OR_LSTM:
        net4=TemporalConvNet(TCN_NUM,TCN_HIDDEN_NUM)

    else:
        net4=BILSTM(HIDDEN_NUM,HIDDEN_NUM,2,device)

    net4 = net4.to(device)

    net5 = Regressor(HIDDEN_NUM,HIDDEN_NUM)
    net5 = net5.to(device)


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

    for epoch in range(pre_epoch, EPOCH):

        optimizer = optim.Adam(itertools.chain(net4.parameters(),net5.parameters()), lr=LR,weight_decay=WEIGHT_DELAY)
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
            y=y.to(device)
            feas=[]
            
            optimizer.zero_grad()
            for x in xs:
                with torch.no_grad():
                    x = x.to(device)
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
        net4.eval()
        net5.eval()
        with torch.no_grad():
            for i,data in enumerate(test_dataloader):
                xs,npys,y=data.values()
                y=y.to(torch.float32).unsqueeze(1)
                y=y.to(device)
                feas=[]
                
                for x in xs:
                    x = x.to(device)
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

        print('  [Test] Loss: %.03f'
                  % (sum_loss / cnt))

        testloss = sum_loss / cnt
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
    print(testloss_best)


def fvTrain():
    checkpoint1 = torch.load(os.path.join(LOGS_ROOT, 'v4_face_rgb_res.t7'))
    print(checkpoint1["acc"])
    checkpoint2 = torch.load(os.path.join(LOGS_ROOT, 'v4_voice_res.t7'))
    print(checkpoint2["acc"])

    resnet_face=torchvision.models.resnet18(pretrained=True)
    net2_face=nn.Sequential(*list(resnet_face.children())[:-1])
    net2_face.load_state_dict(checkpoint1['net2'])
    net2_face = net2_face.to(device)

    resnet_voice=torchvision.models.resnet18(pretrained=True)
    resnet_voice.conv1= nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
    net2_voice=nn.Sequential(*list(resnet_voice.children())[:-1])
    net2_voice.load_state_dict(checkpoint2['net2'])
    net2_voice = net2_voice.to(device)

    net3_face = Regressor(EXTRACT_NUM,HIDDEN_NUM)
    net3_face.load_state_dict(checkpoint1['net3'])
    net3_face = net3_face.to(device)
    
    net3_voice = Regressor(EXTRACT_NUM,HIDDEN_NUM)
    net3_voice.load_state_dict(checkpoint2['net3'])
    net3_voice = net3_voice.to(device)

    if not TCN_OR_LSTM:
        net4=TemporalConvNet(TCN_NUM,TCN_HIDDEN_NUM)

    else:
        net4=BILSTM(HIDDEN_NUM*2,HIDDEN_NUM*2,2,device)

    net4 = net4.to(device)

    net5 = Regressor(HIDDEN_NUM*2,HIDDEN_NUM*2)
    net5 = net5.to(device)


    criterion = nn.MSELoss() 
    
    train_dataset=FVDataset(1,TRAIN_RIO,DATA_PATHS,tcn_num=TCN_NUM)
    test_dataset=FVDataset(0,TRAIN_RIO,DATA_PATHS,tcn_num=TCN_NUM)

    train_dataloader = DataLoader(train_dataset, batch_size=TCN_BATCH_SIZE,shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=TCN_BATCH_SIZE,shuffle=False)
    testloss_best=1e9

    for epoch in range(pre_epoch, EPOCH):

        optimizer = optim.Adam(itertools.chain(net4.parameters(),net5.parameters()), lr=LR,weight_decay=WEIGHT_DELAY)
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
            y=y.to(device)
            feas_face=[]
            feas_voice=[]
            
            optimizer.zero_grad()
            for x in xs:
                with torch.no_grad():
                    x = x.to(device)
                    x = net2_face(x)
                    if VGG_OR_RESNET:
                        x=x.squeeze(-1).squeeze(-1)
                    _,fea = net3_face(x)
                    feas_face.append(fea)

            for x in npys:
                with torch.no_grad():
                    x = x.to(device)
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
                y=y.to(device)
                feas_face=[]
                feas_voice=[]
                
                for x in xs:
                    x = x.to(device)
                    x = net2_face(x)
                    if VGG_OR_RESNET:
                        x=x.squeeze(-1).squeeze(-1)
                    _,fea = net3_face(x)
                    feas_face.append(fea)

                for x in npys:
                    x = x.to(device)
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

fvtest()