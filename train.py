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
from loader import *
import pdb
import matplotlib.pyplot as plt
from train_module import *

parser = argparse.ArgumentParser(description='PyTorch Biovid PreTraining')
parser.add_argument('--yamlFile', default='config/config.yaml', help='yaml file') 
args = parser.parse_args()

cfg=getCfg(args.yamlFile)

os.environ["CUDA_VISIBLE_DEVICES"] = cfg["GPUID"]
device = "cuda" if torch.cuda.is_available() else "cpu"
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

AU_INPUT_SIZE=cfg["AU_INPUT_SIZE"]
AU_HIDDEN_SIZE=cfg["AU_HIDDEN_SIZE"]
AU_OUTPUT_SIZE=cfg["AU_OUTPUT_SIZE"]

DATA_ROOT=cfg["DATA_ROOT"]
MODEL_ROOT=cfg["MODEL_ROOT"]
LOGS_ROOT=cfg["LOGS_ROOT"]
MODEL_NAME=cfg["MODEL_NAME"]
MODEL_NAME2=cfg["MODEL_NAME2"]
CHECKPOINT_NAME=cfg["CHECKPOINT_NAME"]
CHECKPOINT_NAME2=cfg["CHECKPOINT_NAME2"]
CHECKPOINT_NAME3=cfg["CHECKPOINT_NAME3"]

TRAIN_RIO=cfg["TRAIN_RIO"]
DATA_PATHS=cfg["DATA_PATHS"]
PIC_SIZE=cfg["PIC_SIZE"]
IS_POINT=cfg["IS_POINT"]

def extractor_train(modal):
    dataset=DataSet(BATCH_SIZE,TRAIN_RIO,DATA_PATHS,modal,is_time=False,collate_fn=None,pic_size=PIC_SIZE)
    extractor=Resnet_regressor(modal)
    model=SingleModel(extractor,TCN(512,64,TCN_HIDDEN_NUM),Regressor_self(64,64,64,is_droup=0.6),modal)
    # if modal=='face':
    #     model.load_checkpoint(model.extractor,torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME)))
    model.train_init(dataset,LR,WEIGHT_DELAY,[model.extractor],nn.MSELoss(),nn.L1Loss())
    #model.load_checkpoint(model.extractor,torch.load(os.path.join(LOGS_ROOT,MODEL_NAME)))
    model.extractor_train(EPOCH,os.path.join(LOGS_ROOT,MODEL_NAME))

def classifier_train(modal):
    dataset=DataSet(BATCH_SIZE,TRAIN_RIO,DATA_PATHS,modal,is_time=False,collate_fn=None,pic_size=PIC_SIZE)
    model=SingleModel(Resnet_regressor(modal),TCN(512,64,TCN_HIDDEN_NUM),Regressor_self(64,64,64,is_droup=0.6),modal,classifier=Classifier(EXTRACT_NUM,HIDDEN_NUM,CLASS_NUM))
    model.train_init(dataset,LR,WEIGHT_DELAY,[model.extractor,model.classifier],nn.CrossEntropyLoss(),nn.CrossEntropyLoss())
    model.load_checkpoint(torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME)))
    model.classifier_train(EPOCH,os.path.join(LOGS_ROOT,MODEL_NAME))

def time_extractor_train(modal,is_extractor_train):
    dataset=DataSet(TCN_BATCH_SIZE,TRAIN_RIO,DATA_PATHS,modal,is_time=True,collate_fn=collate_fn,pic_size=PIC_SIZE)
    #extractor=VGG_regressor() if modal=='face' else Resnet_regressor(modal) if modal=='voice' else AU_extractor(AU_INPUT_SIZE,AU_HIDDEN_SIZE,AU_OUTPUT_SIZE)
    model=SingleModel(Resnet_regressor(modal),Time_SelfAttention(EXTRACT_NUM,HIDDEN_NUM),Regressor(HIDDEN_NUM*2,HIDDEN_NUM),modal)
    if not is_extractor_train:
        model.load_checkpoint(torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME)))
    model.train_init(dataset,LR,WEIGHT_DELAY,[model.time_extractor,model.regressor],nn.MSELoss(),nn.L1Loss())
    model.time_extractor_train(EPOCH,os.path.join(LOGS_ROOT,MODEL_NAME),is_extractor_train)

def extractor_test(modal):
    dataset=DataSet(BATCH_SIZE,TRAIN_RIO,DATA_PATHS,modal,is_time=False,collate_fn=None,pic_size=PIC_SIZE)
    #extractor=VGG_regressor() if modal=='face' else Resnet_regressor(modal)
    model=SingleModel(Resnet_regressor(modal),TCN(512,64,TCN_HIDDEN_NUM),Regressor_self(64,64,64,is_droup=0.6),modal)
    model.load_checkpoint(torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME)))
    model.train_init(dataset,LR,WEIGHT_DELAY,[model.extractor],nn.MSELoss(),nn.L1Loss())
    print(model.extractor_test(model.test_criterion))

def voice_train():
    dataset=DataSet(TCN_BATCH_SIZE,TRAIN_RIO,DATA_PATHS,'face',is_time=True,collate_fn=collate_fn,pic_size=PIC_SIZE)
    facemodel=SingleModel(Resnet_regressor('face'),Time_SelfAttention(EXTRACT_NUM,HIDDEN_NUM),Regressor(HIDDEN_NUM*2,HIDDEN_NUM),"face")
    voicemodel=SingleModel(Resnet_regressor('voice'),Time_SelfAttention(EXTRACT_NUM,HIDDEN_NUM),Regressor(HIDDEN_NUM*2,HIDDEN_NUM),"voice")
    model=TwoModel(facemodel,voicemodel,Voice_Time_CrossAttention(EXTRACT_NUM,HIDDEN_NUM))
    model.train_init(dataset,LR,WEIGHT_DELAY,[model.CrossModel,model.VoiceModel.regressor])
    model.load_checkpoint(torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME)),torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME2)))
    model.voice_train(EPOCH,os.path.join(LOGS_ROOT,MODEL_NAME))

def two_train():
    dataset=DataSet(TCN_BATCH_SIZE,TRAIN_RIO,DATA_PATHS,'face',is_time=True,collate_fn=collate_fn,pic_size=PIC_SIZE)
    facemodel=SingleModel(Resnet_regressor('face'),Time_SelfAttention(EXTRACT_NUM,HIDDEN_NUM),Regressor(HIDDEN_NUM*2,HIDDEN_NUM),"face")
    voicemodel=SingleModel(Resnet_regressor('voice'),Time_SelfAttention(EXTRACT_NUM,HIDDEN_NUM),Regressor(HIDDEN_NUM*2,HIDDEN_NUM),"voice")
    model=TwoModel(facemodel,voicemodel,Voice_Time_CrossAttention(EXTRACT_NUM,HIDDEN_NUM),Regressor(HIDDEN_NUM*4,HIDDEN_NUM))
    model.train_init(dataset,LR,WEIGHT_DELAY,[model.regressor])
    model.load_checkpoint(torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME)),torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME2)),torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME3)))
    model.train(EPOCH,os.path.join(LOGS_ROOT,MODEL_NAME2))

voice_train()
two_train()
#time_extractor_train('voice',False)
#extractor_test('face')
