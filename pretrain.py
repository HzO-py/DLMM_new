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
from torch.utils.data import DataLoader
from utils import getCfg
from models import Prototype,Classifier,ResNet18,cnn1d
from loader import BioDataset
import pdb

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='PyTorch Biovid PreTraining')
    parser.add_argument('--yamlFile', default='config.yaml', help='yaml file') 
    args = parser.parse_args()

    cfg=getCfg(args.yamlFile)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["GPUID"]
    EPOCH = cfg["EPOCH"]
    pre_epoch = 0  
    BATCH_SIZE = cfg["BATCH_SIZE"]
    trainacc_best  = 0
    EXTRACT_NUM=cfg["EXTRACT_NUM"]
    CLASS_NUM=cfg["CLASS_NUM"]
    DATA_ROOT=cfg["DATA_ROOT"]
    LOGS_ROOT=cfg["LOGS_ROOT"]
    TRAIN_RIO=cfg["TRAIN_RIO"]
    PROTOTYPE_HIDDEN_NUM=cfg["PROTOTYPE_HIDDEN_NUM"]
    MODAL=cfg["MODAL"]
    
    net1 = Prototype(EXTRACT_NUM,PROTOTYPE_HIDDEN_NUM,CLASS_NUM).to(device)
    net2 = cnn1d(EXTRACT_NUM).to(device)
    net3 = Classifier(EXTRACT_NUM,CLASS_NUM).to(device) 

    criterion = nn.CrossEntropyLoss() 

    net1.train()
    net2.train()
    net3.train()
    
    train_dataset=BioDataset(1,TRAIN_RIO,DATA_ROOT,MODAL)
    test_dataset=BioDataset(0,TRAIN_RIO,DATA_ROOT,MODAL)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,shuffle=True)

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
       
        optimizer = optim.Adam(itertools.chain(net2.parameters(),net1.parameters(),net3.parameters()), lr=LR)
        print('\nEpoch: %d' % (epoch + 1))
        
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        acc = 0
        
        for i,data in tqdm(enumerate(train_dataloader)):
            x,y=data.values()

            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            
            #xx = net2(x)
            outputs = net3(x)
            

            out2,fc_w1,fc_w2 = net1(x)
            w = fc_w2[0]
            fc_loss_batch = 0
            #pdb.set_trace()

            for batch_num in range(x.size()[0]):
                for iiiii in range(CLASS_NUM):
                    eu_distance = -1*torch.norm(out2[batch_num,::] - w[iiiii,::].reshape(PROTOTYPE_HIDDEN_NUM)) #负的欧式距离
                    gussian_distance = torch.exp(eu_distance)
                    if iiiii == 0:
                        max_gussian = gussian_distance
                        max_id = 0
                    if max_gussian < gussian_distance:
                        max_gussian = gussian_distance
                        max_id = iiiii
                    if y[batch_num].item() == iiiii:
                        fc_loss = -torch.log(0.0000001+gussian_distance.reshape(1))/2
                       
                    else:
                        fc_loss = -torch.log(0.0000001+1-gussian_distance.reshape(1))/2
                    fc_loss_batch = fc_loss + fc_loss_batch
                if max_id == y[batch_num]:
                    acc = acc +1
            
            
            fc_loss_batch = fc_loss_batch / BATCH_SIZE
            fc_loss_all = fc_loss_batch.cpu().item() + fc_loss_all
            
            loss = criterion(outputs, y)
            
            weight = 0.2
            union_loss = weight*loss + (1-weight)*fc_loss_batch
          
            union_loss.backward()
            optimizer.step()

            # 每训练1个batch打印一次loss和准确率
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += predicted.eq(y.data).cpu().sum()

            # print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
            #       % (epoch + 1, (i + 1 ), sum_loss / (i + 1), 100. * correct / total))
            # print('  loss:',fc_loss_all/(i+1),' acc:',acc/((i+1)*BATCH_SIZE))
        
        torch.save(net1.state_dict(), os.path.join(LOGS_ROOT,MODAL+'_pretrain_net1_%03d.pth' % (epoch + 1)))
        torch.save(net2.state_dict(), os.path.join(LOGS_ROOT,MODAL+'_pretrain_net2_%03d.pth' % (epoch + 1)))
        torch.save(net3.state_dict(), os.path.join(LOGS_ROOT,MODAL+'_pretrain_net3_%03d.pth' % (epoch + 1)))
        trainacc = 100. * correct / total
        if trainacc > trainacc_best:
            trainacc_best = trainacc
            trainacc_best_epoch = epoch+1
            print('trainacc_best:')
            print(trainacc_best)
            
            
            
main()