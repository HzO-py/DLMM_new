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
from models import Prototype,Classifier,ResNet18,cnn1d,VGG
from loader import BioDataset,FaceDataset
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch Biovid PreTraining')
parser.add_argument('--yamlFile', default='bio_config.yaml', help='yaml file') 
args = parser.parse_args()

cfg=getCfg(args.yamlFile)

os.environ["CUDA_VISIBLE_DEVICES"] = cfg["GPUID"]
EPOCH = cfg["EPOCH"]
pre_epoch = 0  
BATCH_SIZE = cfg["BATCH_SIZE"]
testacc_best  = 0
EXTRACT_NUM=cfg["EXTRACT_NUM"]
CLASS_NUM=cfg["CLASS_NUM"]
DATA_ROOT=cfg["DATA_ROOT"]
MODEL_ROOT=cfg["MODEL_ROOT"]
LOGS_ROOT=cfg["LOGS_ROOT"]
TRAIN_RIO=cfg["TRAIN_RIO"]
MODAL=cfg["MODAL"]

def test():
    net2 = VGG("VGG19")
    checkpoint = torch.load(os.path.join(LOGS_ROOT, 'face.t7'))
    net2.load_state_dict(checkpoint['net2'])
    net2 = net2.to(device)
    net3 = Classifier(EXTRACT_NUM,CLASS_NUM)
    net3.load_state_dict(checkpoint['net3'])
    net3 = net3.to(device) 
    criterion = nn.CrossEntropyLoss() 
    test_datasets=[FaceDataset(0,0),FaceDataset(0,0.9),FaceDataset(1,0.9)]
    test_dataloaders=[]
    for test_dataset in test_datasets:
        test_dataloaders.append(DataLoader(test_dataset, batch_size=BATCH_SIZE,shuffle=False))
    for test_dataloader in test_dataloaders:
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        acc = 0
        cnt=0
        net2.eval()
        net3.eval()
        for i,data in enumerate(test_dataloader):
            x,y=data.values()

            x, y = x.to(device), y.to(device)

            
            x = net2(x)
            outputs = net3(x)
            
            loss = criterion(outputs, y)
            cnt+=1
            # 每训练1个batch打印一次loss和准确率
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += predicted.eq(y.data).cpu().sum()

        print('  [Test] Loss: %.03f | Acc: %.3f%%'
                  % (sum_loss / cnt, 100. * correct / total))

def main():
    net3 = Classifier(EXTRACT_NUM,CLASS_NUM).to(device) 

    criterion = nn.CrossEntropyLoss() 
    
    train_dataset=BioDataset(1,TRAIN_RIO,MODAL)
    test_dataset=BioDataset(0,TRAIN_RIO,MODAL)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,shuffle=False)

    for epoch in range(pre_epoch, EPOCH):
        fc_loss_all = 0.0
        if epoch < 20:
            LR = 0.001
        elif epoch < 100:
            LR = 0.0001
        elif epoch < 160:
            LR = 0.00001
        else :
            LR =0.00001
       
        optimizer = optim.Adam(net3.parameters(), lr=LR)
        print('\nEpoch: %d' % (epoch + 1))
        
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        acc = 0

        net3.train()
        for i,data in enumerate(train_dataloader):
            x,y=data.values()

            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            #print(x.size())
            outputs = net3(x)
            
            loss = criterion(outputs, y)
          
            loss.backward()
            optimizer.step()

            # 每训练1个batch打印一次loss和准确率
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += predicted.eq(y.data).cpu().sum()
            
            #sys.stdout.flush()
            if i>0:
                sys.stdout.write('\r')
            sys.stdout.write('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%%'
                  % (epoch + 1, (i + 1 ), sum_loss / (i + 1), 100. * correct / total))
            
        
            # print('  loss:',fc_loss_all/(i+1),' acc:',acc/((i+1)*BATCH_SIZE))
        ####################################################################################
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        acc = 0
        cnt=0

        net3.eval()
        for i,data in enumerate(test_dataloader):
            x,y=data.values()

            x, y = x.to(device), y.to(device)

            outputs = net3(x)
            
            loss = criterion(outputs, y)
            cnt+=1
            # 每训练1个batch打印一次loss和准确率
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += predicted.eq(y.data).cpu().sum()

        print('  [Test] Loss: %.03f | Acc: %.3f%%'
                  % (sum_loss / cnt, 100. * correct / total))

        testacc = 100. * correct / total
        global testacc_best
        if testacc > testacc_best:
            testacc_best = testacc
            state = {
            'net3': net3.state_dict(),
            'acc': testacc_best,
            'epoch': epoch,
        }
            torch.save(state, os.path.join(LOGS_ROOT,'gsr.t7'))            
    print(testacc_best)
            

main()