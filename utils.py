import numpy as np
from scipy.fftpack import fft
import os
import yaml
import csv
import pdb

label2num={"BL1":0,"PA1":1,"PA2":2,"PA3":3,"PA4":4}
modal2num={"gsr":3,"ecg":1}

abc2num={"A":0.0,"B":0.25,"C":0.5,"D":0.75,"E":1.0}

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
        if np.isnan(fTemp).any():
            print( fTemp, fileName)
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

def getLable(label_path,person,video_id):
    with open(label_path, 'r',encoding="utf-8") as f:
        reader = csv.reader(f)
        lines=[line for line in reader][1:]
        for line in lines:
            if line[0]==person:
                if video_id==-1:
                    if line[1]!="":
                        return True,float(int(line[1])/10)
                    score=0.0
                    flag=False
                    if line[2]!="":
                        flag=True
                        score+=float(int(line[2])/10)
                    if line[3]!="":
                        flag=True
                        if line[3] in ["A","B","C","D","E"]:
                            score+=float(abc2num[line[3]])
                        else:
                            score+=float((int(line[3])-1)*0.25)
                    if flag:
                        return True,score/2
                elif video_id==2 or video_id==4:
                    base=1
                    if video_id==4:
                        base=5
                    score=0.0
                    flag=False
                    if line[base]!="":
                        flag=True
                        score+=float(int(line[base])/10)
                    if line[base+1]!="":
                        flag=True
                        score+=float(int(line[base+1])/10)
                    if flag:
                        return True,score/2
                    score=0.0
                    flag=False
                    if line[base+2]!="":
                        flag=True
                        score+=float(int(line[base+2])/10)
                    if line[base+3]!="":
                        flag=True
                        if line[base+3] in ["A","B","C","D","E"]:
                            score+=float(abc2num[line[base+3]])
                        else:
                            score+=float((int(line[base+3])-1)*0.25)
                    if flag:
                        return True,score/2
    return False,False

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
                    sample.append([os.path.join(root_path_3,img),int(label)])
        if len(sample)>0:
            samples.append(sample)
    
    return samples 

def getBioSample(root_path,label_path):
    samples=[]
    for person in sorted(os.listdir(root_path),key=lambda x:int(x.split('-')[0])):
        ret,label=getLable(label_path,person.split('-')[0])
        if not ret:
            continue
        root_path_2=os.path.join(root_path,person)
        with open(root_path_2, 'r') as f:
            reader = csv.reader(f)
            lines=[line for line in reader]
            if len(lines)<5:
                continue
        samples.append([root_path_2,int(label)])
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
