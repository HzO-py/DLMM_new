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
MODAL=cfg["MODAL"]


VGG_OR_RESNET=cfg["VGG_OR_RESNET"]
EXTRACT_NUM=cfg["EXTRACT_NUM"]
EXTRACT_NUM=3 if MODAL=='bio' else 512
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
CLUSTER_ROOT=cfg["CLUSTER_ROOT"]

MODEL_NAME=cfg["MODEL_NAME"]
MODEL_NAME2=cfg["MODEL_NAME2"]
CHECKPOINT_NAME=cfg["CHECKPOINT_NAME"]
CHECKPOINT_NAME2=cfg["CHECKPOINT_NAME2"]
CHECKPOINT_NAME3=cfg["CHECKPOINT_NAME3"]
CHECKPOINT_NAME4=cfg["CHECKPOINT_NAME4"]

TRAIN_RIO=cfg["TRAIN_RIO"]
DATA_PATHS=cfg["DATA_PATHS"]
PIC_SIZE=cfg["PIC_SIZE"]
IS_POINT=cfg["IS_POINT"]

SAMPLE_THRESHOLD=cfg["SAMPLE_THRESHOLD"]
SCORE_THRESHOLD=cfg["SCORE_THRESHOLD"]
CLUSTER_NUM=cfg["CLUSTER_NUM"]

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(cfg["SEED"])

def extractor_train(modal):
    dataset=DataSet(BATCH_SIZE,TRAIN_RIO,DATA_PATHS,modal,is_time=False,collate_fn=None,pic_size=PIC_SIZE)
    extractor=Resnet_regressor(modal)
    model=SingleModel(extractor,Time_SelfAttention(EXTRACT_NUM,HIDDEN_NUM),Regressor(HIDDEN_NUM*2,HIDDEN_NUM),modal)
    model.train_init(dataset,LR,WEIGHT_DELAY,[model.extractor],nn.MSELoss(),nn.L1Loss())
    model.load_checkpoint(torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME)))
    model.extractor_train(EPOCH,os.path.join(LOGS_ROOT,MODEL_NAME))

def classifier_train(modal):
    dataset=DataSet(BATCH_SIZE,TRAIN_RIO,DATA_PATHS,modal,is_time=False,collate_fn=None,pic_size=PIC_SIZE)
    model=SingleModel(Resnet_regressor(modal),TCN(512,64,TCN_HIDDEN_NUM),Regressor_self(64,64,64,is_droup=0.6),modal,classifier=Classifier(EXTRACT_NUM,HIDDEN_NUM,CLASS_NUM))
    model.train_init(dataset,LR,WEIGHT_DELAY,[model.extractor,model.classifier],nn.CrossEntropyLoss(),nn.CrossEntropyLoss())
    model.load_checkpoint(torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME)))
    model.classifier_train(EPOCH,os.path.join(LOGS_ROOT,MODEL_NAME))

def time_extractor_train(modal,is_selfatt):
    dataset=DataSet(TCN_BATCH_SIZE,TRAIN_RIO,DATA_PATHS,modal,is_time=True,collate_fn=None,pic_size=PIC_SIZE)
    extractor=NoChange() if modal=='bio' else Resnet_regressor(modal)
    model=SingleModel(extractor,Time_SelfAttention(EXTRACT_NUM,HIDDEN_NUM),Regressor(HIDDEN_NUM*2,HIDDEN_NUM),modal)
    # if modal!='bio':
    #     model.load_checkpoint(torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME)))
    model.train_init(dataset,LR,WEIGHT_DELAY,[model.time_extractor,model.regressor],nn.MSELoss(),nn.L1Loss())
    model.load_time_checkpoint(torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME)))
    model.time_extractor_train(EPOCH,os.path.join(LOGS_ROOT,MODEL_NAME),is_selfatt=is_selfatt)

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
    model.load_checkpoint(torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME)),torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME2)),torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME3)),torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME4)))
    model.train(EPOCH,os.path.join(LOGS_ROOT,MODEL_NAME))

def three_train():
    dataset=DataSet(TCN_BATCH_SIZE,TRAIN_RIO,DATA_PATHS,'bio',is_time=True,collate_fn=collate_fn,pic_size=PIC_SIZE)
    facemodel=SingleModel(Resnet_regressor('face'),Time_SelfAttention(EXTRACT_NUM,HIDDEN_NUM),Regressor(HIDDEN_NUM*2,HIDDEN_NUM),"face")
    voicemodel=SingleModel(Resnet_regressor('voice'),Time_SelfAttention(EXTRACT_NUM,HIDDEN_NUM),Regressor(HIDDEN_NUM*2,HIDDEN_NUM),"voice")
    biomodel=BioModel(Time_SelfAttention(3,HIDDEN_NUM),Regressor(HIDDEN_NUM*2,HIDDEN_NUM),'bio')
    model=TwoModel(facemodel,voicemodel,Voice_Time_CrossAttention(EXTRACT_NUM,HIDDEN_NUM),Regressor_self_att(3,HIDDEN_NUM//2,HIDDEN_NUM//2,0.1),biomodel)
    model.train_init(dataset,LR,WEIGHT_DELAY,[model.regressor])
    model.load_checkpoint(torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME)),torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME2)),bio_checkout=torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME3)))
    model.train(EPOCH,os.path.join(LOGS_ROOT,MODEL_NAME))
    
def cluster_train(modal,is_selfatt,checkpoint,pre_model_id,model_id,pre_score,save_cluster,cluster_num):
    dataset=DataSet(TCN_BATCH_SIZE,TRAIN_RIO,DATA_PATHS,modal,is_time=True,collate_fn=collate_fn,pic_size=PIC_SIZE)
    extractor=NoChange() if modal=='bio' else Resnet_regressor(modal)
    cluster=ClusterCenter(HIDDEN_NUM*2,cluster_num)
    model=SingleModel(extractor,Time_SelfAttention(EXTRACT_NUM,HIDDEN_NUM),Regressor(HIDDEN_NUM*2,HIDDEN_NUM),modal,cluster=cluster)
    model.load_time_checkpoint(checkpoint["modelList"][pre_model_id])
    model.train_init(dataset,LR,WEIGHT_DELAY,[model.cluster],nn.MSELoss(),nn.L1Loss())
    return model.feature_space(is_selfatt=is_selfatt,savepath=os.path.join(save_cluster,str(pre_model_id)),pre_space_path=checkpoint['space_path'],pre_model_id=pre_model_id,model_id=model_id,sample_threshold=SAMPLE_THRESHOLD,score_threshold=SCORE_THRESHOLD,pre_score=pre_score,cluster_num=cluster_num)

def ori_MultiExperts_train(modal):
    #space_path,centerList=cluster_train(modal,True)

    dataset=DataSet(TCN_BATCH_SIZE,TRAIN_RIO,DATA_PATHS,modal,is_time=True,collate_fn=collate_fn,pic_size=PIC_SIZE)
    extractor=NoChange() if modal=='bio' else Resnet_regressor(modal)
    modelList=[]
    checkList=[]
    for _ in range(3):
        modelList.append(SingleModel(extractor,Time_SelfAttention(EXTRACT_NUM,HIDDEN_NUM),Regressor(HIDDEN_NUM*2,HIDDEN_NUM),modal))
        checkList.append(torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME)))
    experts=MultiExperts(modelList)
    experts.load_checkpoint(checkList,torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME)))
    experts.train_init(dataset,LR,WEIGHT_DELAY)
    experts.train(EPOCH,os.path.join(LOGS_ROOT,MODEL_NAME))

def MultiExperts_train(modal):
    dataset=DataSet(TCN_BATCH_SIZE,TRAIN_RIO,DATA_PATHS,modal,is_time=True,collate_fn=collate_fn,pic_size=PIC_SIZE)

    checkpoint=torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME))
    checkpoint["modelList"]=[{"extractor":checkpoint["extractor"],"time_extractor":checkpoint["time_extractor"],"regressor":checkpoint["regressor"]}]
    checkpoint["score"]=1e3
    checkpoint['space_path']={}

    cnt=0

    while 1:
        if cnt>=1:
            break

        print("---------------------Generation:",cnt)
        space_path={}  
        modelList=[]
        checkList=[]
        save_model=os.path.join(LOGS_ROOT,MODEL_NAME+str(cnt)+"G.t7")
        save_cluster=os.path.join(CLUSTER_ROOT,MODEL_NAME+str(cnt)+"G.t7")

        for i in range(len(checkpoint["modelList"])):
            space_path_sub,num,centerList,stdList=cluster_train(modal,is_selfatt=True,checkpoint=checkpoint,pre_model_id=i,model_id=len(modelList),pre_score=checkpoint["score"],save_cluster=save_cluster,cluster_num=CLUSTER_NUM)
            space_path.update(space_path_sub)
            for _ in range(num):
                extractor=NoChange() if modal=='bio' else Resnet_regressor(modal)
                modelList.append(SingleModel(extractor,Time_SelfAttention(EXTRACT_NUM,HIDDEN_NUM),Regressor(HIDDEN_NUM*2,HIDDEN_NUM),modal))
                checkList.append(checkpoint["modelList"][i])

        if len(modelList)==len(checkpoint["modelList"]):
            break
        
        experts=MultiExperts(modelList)
        experts.load_checkpoint(checkList,space_path=space_path,centerList=centerList,stdList=stdList)
        experts.train_init(dataset,LR,WEIGHT_DELAY)
        experts.train(EPOCH,save_model)
        checkpoint=torch.load(save_model)
        cnt+=1

def MultiExperts_checkpoint_train(modal):
    dataset=DataSet(TCN_BATCH_SIZE,TRAIN_RIO,DATA_PATHS,modal,is_time=True,collate_fn=collate_fn,pic_size=PIC_SIZE)

    checkpoint=torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME))

    modelList=[]
    checkList=[]
    save_model=os.path.join(LOGS_ROOT,CHECKPOINT_NAME)

    for i in range(len(checkpoint["modelList"])):
        extractor=NoChange() if modal=='bio' else Resnet_regressor(modal)
        modelList.append(SingleModel(extractor,Time_SelfAttention(EXTRACT_NUM,HIDDEN_NUM),Regressor(HIDDEN_NUM*2,HIDDEN_NUM),modal))
        checkList.append(checkpoint["modelList"][i])
    
    experts=MultiExperts(modelList)
    experts.load_checkpoint(checkList,space_path=checkpoint["space_path"],centerList=checkpoint["centerList"],stdList=checkpoint["stdList"])
    experts.train_init(dataset,LR,WEIGHT_DELAY)
    experts.testloss_best=checkpoint["acc"]
    print(checkpoint["acc"])
    experts.train(EPOCH,save_model)

def MultiExperts_test(modal,modelNum):
    dataset=DataSet(TCN_BATCH_SIZE,TRAIN_RIO,DATA_PATHS,modal,is_time=True,collate_fn=collate_fn,pic_size=PIC_SIZE)
    modelList=[]
    for _ in range(modelNum):
        modelList.append(SingleModel(Resnet_regressor(modal),Time_SelfAttention(EXTRACT_NUM,HIDDEN_NUM),Regressor(HIDDEN_NUM*2,HIDDEN_NUM),modal))
    backbone=SingleModel(Resnet_regressor(modal),Time_SelfAttention(EXTRACT_NUM,HIDDEN_NUM),Regressor(HIDDEN_NUM*2,HIDDEN_NUM),modal)
    experts=MultiExperts(modelList,backbone)
    experts.test_init(torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME)),dataset)
    experts.test()

def Mul_MultiExperts_test(modelNum):
    dataset=DataSet(TCN_BATCH_SIZE,TRAIN_RIO,DATA_PATHS,"bio",is_time=True,collate_fn=collate_fn,pic_size=PIC_SIZE)
    modelList=[]
    for _ in range(modelNum):
        modelList.append(SingleModel(Resnet_regressor("face"),Time_SelfAttention(EXTRACT_NUM,HIDDEN_NUM),Regressor(HIDDEN_NUM*2,HIDDEN_NUM),"face"))
    for _ in range(modelNum):
        modelList.append(SingleModel(Resnet_regressor("voice"),Time_SelfAttention(EXTRACT_NUM,HIDDEN_NUM),Regressor(HIDDEN_NUM*2,HIDDEN_NUM),"voice"))
    # for _ in range(modelNum):
    #     modelList.append(SingleModel(NoChange(),Time_SelfAttention(3,HIDDEN_NUM),Regressor(HIDDEN_NUM*2,HIDDEN_NUM),"bio"))
    experts=MultiExperts(modelList)
    checkpointList=[torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME)),torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME2))]
    experts.mul_test_init(checkpointList,dataset)
    experts.test()

#voice_train()
#three_train()
#time_extractor_train(MODAL,is_selfatt=True)
#bio_train(BIO_MODAL,is_selfatt=True,is_pro=True) 
#extractor_test(MODAL)
#extractor_train(MODAL)

#print(torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME))["acc"],torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME))["test_hunxiao"])
#cluster_train(MODAL,is_selfatt=True)
MultiExperts_train(MODAL)
#MultiExperts_checkpoint_train(MODAL)
#MultiExperts_test(MODAL,3)
#Mul_MultiExperts_test(3)
#MultiExperts_checkpoint_train(MODAL)
#train_dataset=FaceDataset(1,TRAIN_RIO,DATA_PATHS,MODAL,is_time=False,pic_size=PIC_SIZE)