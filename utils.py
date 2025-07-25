import numpy as np
from scipy.fftpack import fft
import os
import yaml
import csv
import pdb
import math
import scipy.linalg as linalg

label2num={"BL1":0,"PA1":1,"PA2":2,"PA3":3,"PA4":4}
modal2num={"gsr":3,"ecg":1}

abc2num={"A":0.0,"B":0.2,"C":0.4,"D":0.6,"E":0.8,"F":1.0}

doctorVSnurse=[]

ages_distribute=[[0,0],[0,0],[0,0],[0,0],[0,0]]
ages_distribute_bio=[0,0,0,0,0]
score_distribute=[[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]]
score_distribute_bio=[[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]]


def spectralCentroid(X):
    """Computes spectral centroid of frame (given abs(FFT))"""
    L = X.shape[0]
    ind = (np.arange(1, len(X) + 1)) * (100/(2.0 * len(X)))
    Xt = X.copy()
    Xt = Xt / Xt.max()
    NUM = np.sum(ind * Xt)
    DEN = np.sum(Xt) + 0.000000001
    # Centroid:
    C = (NUM / DEN)
    return C

def stSpectralRollOff(X, c):
    """Computes spectral roll-off"""
    totalEnergy = np.sum(X ** 2)
    fftLength = len(X)
    Thres = c*totalEnergy
    # Ffind the spectral rolloff as the frequency position where the respective spectral energy is equal to c*totalEne    rgy
    CumSum = np.cumsum(X ** 2) + 0.00000001
    [a, ] = np.nonzero(CumSum > Thres)
    if len(a) > 0:
        mC = np.float64(a[0]) / (float(fftLength))
    else:
        mC = 0.0
    return (mC)

def fileFeatureExtraction(rawData):
    
    rawData = np.nan_to_num(rawData)


    means = rawData.mean(axis = 0)             # compute average
    stds = rawData.std(axis = 0)               # compute std
    maxs = rawData.max(axis = 0)               # compute max values
    mins = rawData.min(axis = 0)               # compute min values
    centroid = []
    rolloff = []

         # compute spectral features
    fTemp = abs(fft(rawData))         # compute FFT
    fTemp = fTemp[0:int(fTemp.shape[0]/2)]  # get the first symetrical FFT part
    c = 0.9999
    if len(fTemp) == 0:
        pass
    else:
        
        centroid=spectralCentroid(fTemp)    # compute spectral centroid
        rolloff=stSpectralRollOff(fTemp, c) # compute spectral rolloff
    #pdb.set_trace()
    #if len(means) == 4 and len(stds) ==4 and len(maxs) == 4 and len(mins)== 4 and len(centroid) == 4 and len(rolloff) ==4:
    featureVector = np.array([means, stds, maxs, mins, centroid, rolloff])  # concatenate features to form the final feature vector
    return featureVector

def getCfg(yaml_path):
    with open(yaml_path,"r",encoding="utf-8") as f:
        cfg=f.read()
        cfg=yaml.load(cfg,Loader=yaml.FullLoader)
        return cfg

def readCsv(modal,csv_path):
    seq=[]
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        lines=[line for line in reader]
        for line in lines:
            line=line[0].split(';')
            seq.append(float(line[modal2num[modal]]))
    return seq

def getSample(root_path):
    samples=[]
    for person in os.listdir(root_path):
        for sample in os.listdir(os.path.join(root_path,person)):
            samples.append([os.path.join(os.path.join(root_path,person),sample),label2num[sample.split('-')[1]]])
    return samples

def getLable(label_path,video,version):
    with open(label_path, 'r',encoding="gbk") as f:
        reader = csv.reader(f)
        lines=[line for line in reader][1:]
        for line in lines:
            if line[2]==video and line[version]!='':
                score=float(line[version])/10.0
                if line[3]!='':
                    return True,score,line[3]
                return True,score,False
    return False,False,False

def getFaceSample(root_path,label_path,version):
    samples=[]
    for person in sorted(os.listdir(root_path),key=lambda x:int(x)):
        sample=[]
        root_path_2=os.path.join(root_path,person)
        for video in os.listdir(root_path_2):
            video_id=-1
            if version==2:
                video_id=int(video.split('.')[0].split('-')[-1])
            ret,label=getLable(label_path,person,video_id)
            if not ret:
                continue
            root_path_3=os.path.join(root_path_2,video)
            for img in sorted(os.listdir(root_path_3),key=lambda x:int(x.split('.')[0])):
                if img.endswith("jpg"):
                    npy=img.split('.')[0]+'.npy'
                    sample.append([os.path.join(root_path_3,img),os.path.join(root_path_3,npy),label])
        if len(sample)>0:
            samples.append(sample)
    
    return samples 

def getAllSample(root_path,label_path,version):
    samples=[]
    for person in sorted(os.listdir(os.path.join(root_path,'face')),key=lambda x:int(x)):
        root_path_2=os.path.join(os.path.join(root_path,'face'),person)
        for video in os.listdir(root_path_2):
            ret,label,bioret=getLable(label_path,video,version)#version==4 or 13
            if not ret:
                continue
            root_path_3=os.path.join(root_path_2,video)
            if len(os.listdir(root_path_3))>0:
                #找同标本不同模态voice
                splits=root_path_3.split('/')
                splits[0]='/'+splits[0]
                splits[-3]='voice'
                sample_name=splits[-1].split('.')[0]
                splits[-1]=sample_name+'.wav_fftnpy'
                voicepath=os.path.join(*splits)
                if os.path.exists(voicepath) and len(os.listdir(voicepath))>0:
                    #bio
                    if bioret:
                        biopath=os.path.join(os.path.join(root_path,'bio'),bioret)
                        samples.append([root_path_3,voicepath,biopath,label])
                    else:
                        samples.append([root_path_3,voicepath,label])
    return samples

def getBioSample(root_path,label_path,version):
    samples=[]
    if not os.path.exists(root_path):
        return samples 
    for person in sorted(os.listdir(root_path),key=lambda x:int(x.split('-')[0])):
        bio_id=-1
        if version==2:
            bio_id=int(person.split('.')[0].split('-')[-1])
        ret,label=getLable(label_path,person.split('-')[0],bio_id)
        if not ret:
            continue
        root_path_2=os.path.join(root_path,person)
        with open(root_path_2, 'r') as f:
            reader = csv.reader(f)
            lines=[line for line in reader]
            if len(lines)<10:
                continue
        samples.append([root_path_2,int(label)])
    return samples 

def getVoiceSample(root_path,label_path,version):
    samples=[]
    for person in sorted(os.listdir(root_path),key=lambda x:int(x)):
        sample=[]
        root_path_2=os.path.join(root_path,person)
        for video in os.listdir(root_path_2):
            if not video.endswith('_fftnpy'):
                continue
            video_id=-1
            if version==2:
                video_id=int(video.split('.')[0].split('-')[-1])
            ret,label=getLable(label_path,person,video_id)
            if not ret:
                continue
            root_path_3=os.path.join(root_path_2,video)
            for wav in sorted(os.listdir(root_path_3),key=lambda x:int(x.split('.')[0])):
                if wav.endswith("npy"):
                    sample.append([os.path.join(root_path_3,wav),label])
        if len(sample)>0:
            samples.append(sample)
    
    return samples 

def handFeature(input_data):
    #input_data=np.array(input_data)
    return np.mean(input_data),np.var(input_data),np.mean(np.abs(input_data)),np.max(input_data)-np.min(input_data),np.std(input_data)


def extractGsr(seq):

    dseq = seq[1:] - seq[:-1]
    # 滤波前二阶导数
    ddseq = dseq[1:] - dseq[:-1]
    # 滤波前三阶导数
    dddseq = ddseq[1:] - ddseq[:-1]
    

    fea=[handFeature(seq),handFeature(dseq),handFeature(ddseq),handFeature(dddseq)]
    return np.array(fea)

def rotate_mat(axis, radian):
    rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
    return rot_matrix

def rotate(a,b,axi,dim):
    inf=1e-9
    acos=a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b)+inf)
    angle=0
    if acos>1.0:
        angle=0
    elif acos<-1.0:
        angle=np.pi
    else:
        angle=np.arccos(acos)
    if a[dim]>0:
        angle=-angle
    rot_matrix=rotate_mat(axi,angle)
    return rot_matrix

def npyStandard(npy):
    
    npy[:]-=npy[30]
    snor=np.linalg.norm(npy[8]-npy[30])
    npy/=snor
    
    a=npy[27].copy()
    a[0]=0
    rot_matrix=rotate(a,np.array([0,0,1]),[1,0,0],1)
    npy=np.dot(npy,rot_matrix)
    
    a=npy[27].copy()
    a[1]=0
    rot_matrix=rotate(a,np.array([1,0,0]),[0,1,0],2)
    npy=np.dot(npy,rot_matrix)
    
    a=(npy[8]-npy[30]).copy()
    a[0]=0
    rot_matrix=rotate(a,np.array([0,0,1]),[1,0,0],1)
    npy=np.dot(npy,rot_matrix)
    
    npy=npy[17:]
    
    return npy

def name():
    paths=[#1-843
    ["/hdd/sda/lzq/DLMM_new/dataset/2022.1.29/pain2","/hdd/sda/lzq/DLMM_new/dataset/2022.1.29/pain2/label.csv",1],
    ["/hdd/sda/lzq/DLMM_new/dataset/2022.2.25/pain1","/hdd/sda/lzq/DLMM_new/dataset/2022.2.25/pain1/label.csv",1],
    ["/hdd/sda/lzq/DLMM_new/dataset/2022.2.25/pain2","/hdd/sda/lzq/DLMM_new/dataset/2022.2.25/pain2/label.csv",1],
    ["/hdd/sda/lzq/DLMM_new/dataset/2022.2.25/pain3","/hdd/sda/lzq/DLMM_new/dataset/2022.2.25/pain3/label.csv",1],
    ["/hdd/sda/lzq/DLMM_new/dataset/2022.2.25/pain4","/hdd/sda/lzq/DLMM_new/dataset/2022.2.25/pain4/label.csv",2],
    ["/hdd/sda/lzq/DLMM_new/dataset/2022.3.5/pain3","/hdd/sda/lzq/DLMM_new/dataset/2022.3.5/label.csv",2],
    ["/hdd/sda/lzq/DLMM_new/dataset/2022.3.5/pain4","/hdd/sda/lzq/DLMM_new/dataset/2022.3.5/label.csv",2],
    #["/hdd/sda/lzq/DLMM_new/dataset/2022.3.5/pain5","/hdd/sda/lzq/DLMM_new/dataset/2022.3.5/label.csv",2],
    ["/hdd/sda/lzq/DLMM_new/dataset/2022.3.23/pain1","/hdd/sda/lzq/DLMM_new/dataset/2022.3.23/label.csv",2],
    ["/hdd/sda/lzq/DLMM_new/dataset/2022.3.23/pain2","/hdd/sda/lzq/DLMM_new/dataset/2022.3.23/label.csv",2],
    ["/hdd/sda/lzq/DLMM_new/dataset/2022.3.23/pain3","/hdd/sda/lzq/DLMM_new/dataset/2022.3.23/label.csv",2],
    ["/hdd/sda/lzq/DLMM_new/dataset/2022.3.23/pain4","/hdd/sda/lzq/DLMM_new/dataset/2022.3.23/label.csv",2],
    ["/hdd/sda/lzq/DLMM_new/dataset/2022.3.23/pain5","/hdd/sda/lzq/DLMM_new/dataset/2022.3.23/label.csv",2],
    ["/hdd/sda/lzq/DLMM_new/dataset/2022.3.23/pain7","/hdd/sda/lzq/DLMM_new/dataset/2022.3.23/label.csv",2],
    ["/hdd/sda/lzq/DLMM_new/dataset/2022.3.23/pain8","/hdd/sda/lzq/DLMM_new/dataset/2022.3.23/label.csv",2]
    ]
    samples=[]
    for path in paths:
        #getBioSample(os.path.join(path[0],"bio"),path[1],path[2])
        samples+=getAllSample(*path)
    print(score_distribute_bio,score_distribute)