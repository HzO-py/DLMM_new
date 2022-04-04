# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 08:18:26 2020

@author: 98669
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
import itertools
from tqdm import tqdm
import os
import sys
from torch.utils.data import DataLoader
from utils import getCfg
from models import Prototype,Classifier,ResNet18,cnn1d,VGG,Regressor,TemporalConvNet,VoiceCNN
from loader import BioDataset,FaceDataset,VoiceDataset
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch Biovid PreTraining')
parser.add_argument('--yamlFile', default='config/voice_config.yaml', help='yaml file') 
args = parser.parse_args()

cfg=getCfg(args.yamlFile)

os.environ["CUDA_VISIBLE_DEVICES"] = cfg["GPUID"]
EPOCH = cfg["EPOCH"]
pre_epoch = 0  
BATCH_SIZE = cfg["BATCH_SIZE"]
LR=cfg["LR"]
WEIGHT_DELAY=cfg["WEIGHT_DELAY"]

EXTRACT_NUM=cfg["EXTRACT_NUM"]
HIDDEN_NUM=cfg["HIDDEN_NUM"]
CLASS_NUM=cfg["CLASS_NUM"]
TCN_NUM=cfg["TCN_NUM"]
TCN_HIDDEN_NUM=cfg["TCN_HIDDEN_NUM"]

DATA_ROOT=cfg["DATA_ROOT"]
MODEL_ROOT=cfg["MODEL_ROOT"]
LOGS_ROOT=cfg["LOGS_ROOT"]
MODEL_NAME=cfg["MODEL_NAME"]

TRAIN_RIO=cfg["TRAIN_RIO"]
DATA_PATHS=cfg["DATA_PATHS"]
IS_POINT=cfg["IS_POINT"]

def test():
    net2 = VGG("VGG19")
    checkpoint = torch.load(os.path.join(LOGS_ROOT, MODEL_NAME))
    print(checkpoint["acc"])
    net2.load_state_dict(checkpoint['net2'])
    net2 = net2.to(device)
    net3 = Regressor(EXTRACT_NUM,HIDDEN_NUM)
    net3.load_state_dict(checkpoint['net3'])
    net3 = net3.to(device) 

    net2_cls = VGG("VGG19")
    checkpoint = torch.load(os.path.join("/hdd/sdd/lzq/DLMM_new/model/logs/face_cls_2.t7"))
    print(checkpoint["acc"])
    net2_cls.load_state_dict(checkpoint['net2'])
    net2_cls = net2_cls.to(device)
    net3_cls = Classifier(512,64,2)
    net3_cls.load_state_dict(checkpoint['net3'])
    net3_cls = net3_cls.to(device)


    test_datasets=[FaceDataset(0,0,DATA_PATHS,is_person=1,tcn_num=TCN_NUM)]

    test_dataloaders=[]
    for test_dataset in test_datasets:
        test_dataloaders.append(DataLoader(test_dataset, batch_size=1,shuffle=False))
    for test_dataloader in test_dataloaders:
        sum_loss = 0.0
        cnt=0
        maxn=0
        minn=1e9
        pre=0.0
        ii=0
        net2.eval()
        net3.eval()
        with torch.no_grad():
            for i,data in enumerate(test_dataloader):
                xs,npys,y=data.values()
                y=float(y)
                pre=0.0
                
        #         for x in xs:
        #             x = x.to(device)

        #             x1 = net2_cls(x)
        #             outputs1 = net3_cls(x1)
        #             _, predicted = torch.max(outputs1.data, 1)
        #             predicted=int(predicted.cpu())

                    
        #             x2 = net2(x)
        #             outputs2 = net3(x2)
        #             score=float(outputs2.data.cpu()[0])

        #             if score>0.5 and predicted==0:
        #                 continue

        #             pre+=score
        #         pre/=len(xs)
        #         mae=abs(pre-y)
        #         if mae>0.1:      
        #             print(pre,y,mae)
        #         sum_loss+=mae
        #         cnt+=1
        # print('  [Test] Loss: %.03f'
        #           % (sum_loss / cnt))

def vggTrain():
    net2 = VGG("VGG19")
    checkpoint = torch.load(os.path.join(MODEL_ROOT, 'PrivateTest_model.t7'))
    net2.load_state_dict(checkpoint['net'])
    net2 = net2.to(device)
    for para in net2.named_parameters():
        if para[0].startswith("features"):
            if int(para[0].split(".")[1])<26:
                para[1].requires_grad = False

    net3 = Regressor(EXTRACT_NUM,HIDDEN_NUM).to(device) 

    criterion = nn.MSELoss() 
    
    train_dataset=FaceDataset(1,TRAIN_RIO,DATA_PATHS,is_person=0)
    test_dataset=FaceDataset(0,TRAIN_RIO,DATA_PATHS,is_person=0)
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
            outputs = net3(x)
            
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
                outputs = net3(x)
                
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
    net2 = torch.load(os.path.join(MODEL_ROOT, 'CNN.ckpt'))
    net2 = net2.to(device)
    
    net4=TemporalConvNet(TCN_NUM,TCN_HIDDEN_NUM)
    net4 = net4.to(device)

    net5 = Regressor(EXTRACT_NUM,EXTRACT_NUM//2)
    net5 = net5.to(device)

    criterion = nn.MSELoss() 
    
    train_dataset=VoiceDataset(1,TRAIN_RIO,DATA_PATHS,is_person=1,tcn_num=TCN_NUM)
    test_dataset=VoiceDataset(0,TRAIN_RIO,DATA_PATHS,is_person=1,tcn_num=TCN_NUM)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,shuffle=False)
    testloss_best=1e9

    for epoch in range(pre_epoch, EPOCH):

        optimizer = optim.Adam(itertools.chain(net4.parameters(),net5.parameters()), lr=LR,weight_decay=WEIGHT_DELAY)
        print('\nEpoch: %d' % (epoch + 1))
        
        sum_loss = 0.0

        net2.eval()
        net4.train()
        net5.train()

        for i,data in enumerate(train_dataloader):
            xs,y=data.values()
            y=y.to(torch.float32).unsqueeze(1)
            y = y + 0.05*torch.randn(y.size()[0],1)
            y=y.to(device)
            feas=[]
            
            optimizer.zero_grad()
            for x in xs:
                with torch.no_grad():
                    x = x.to(device)
                    _,fea = net2(x)
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
        net4.eval()
        net5.eval()
        with torch.no_grad():
            for i,data in enumerate(test_dataloader):
                xs,y=data.values()
                y=y.to(torch.float32).unsqueeze(1)
                y=y.to(device)
                feas=[]
                
                for x in xs:
                    x = x.to(device)
                    _,fea = net2(x)
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
            'net4': net4.state_dict(),
            'net5': net5.state_dict(),
            'acc': testloss_best,
            'epoch': epoch,
        }
            torch.save(state, os.path.join(LOGS_ROOT,MODEL_NAME))            
    print(testloss_best)



tcnTrain()
    