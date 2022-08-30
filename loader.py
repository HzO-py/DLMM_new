from posixpath import split
from cv2 import transform
import torch
import numpy as np
from PIL import Image
from utils import getSample,readCsv,fileFeatureExtraction,getFaceSample,getBioSample,extractGsr,npyStandard,getVoiceSample,getAllSample
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import ToTensor, Resize, RandomCrop,Compose,RandomHorizontalFlip,RandomVerticalFlip,Normalize,ColorJitter
import random
import os
from models import Prototype,Classifier,ResNet18,cnn1d,VGG,Regressor
import torchvision.transforms.functional as tf
from math import *
from torchvision.transforms.functional import rotate

class BioDataset(Dataset):

    def __init__(self,train,train_rio,paths,tcn_num=1):
        self.items=[]
        for path in paths:
            self.items+=getBioSample(*path)
        self.train_rio=int(len(self.items)*train_rio)
        if train:
            self.items=self.items[:self.train_rio]
        else:
            self.items=self.items[self.train_rio:]
        print(len(self.items))

    def __len__(self):
        return len(self.items)

    def get_feature(self, arr):
        return (arr) / (np.max(arr) - np.min(arr))

    def load_seq(self,file_path):
        data = np.loadtxt(file_path, delimiter=";", dtype=np.double)
        seq=np.array([self.get_feature(data[:,1]), self.get_feature(data[:,3]), self.get_feature(data[:,4])])
        return seq

    def __getitem__(self, idr):
        item=self.items[idr]
        x=self.load_seq(item[0])
        sample = {'x': x,'y':int(item[-1])}
        return sample    
        
   
class FaceDataset(Dataset):
    def __init__(self,train,train_rio,paths,is_person,tcn_num=1,person_test=0):
        self.train=train
        self.is_person=is_person
        self.items=[]
        self.person_test=person_test

        for path in paths:
            self.items+=getFaceSample(path[0],path[1],path[2])

        self.train_rio=int(len(self.items)*train_rio)

        transform1=Compose([Resize([108,108]),RandomCrop([96,96]),RandomHorizontalFlip(),RandomVerticalFlip(),ToTensor()])
        transform2=Compose([Resize([96,96]),ToTensor()])


        if train:
            self.items=self.items[:self.train_rio]
            self.transform =transform1
        else:
            self.items=self.items[self.train_rio:]
            self.transform = transform2
        items=[]
        itemss=[]
        
        if not is_person:
            for person in self.items:
                for img in person:
                    items.append(img)
            self.items=items
        else:
            self.transform = transform2
                        
            for person in self.items:
                i=0
                LEN=len(person)

                while i<LEN:
                    item=[]
                    choose_num=[x for x in range(i,i+min(tcn_num,LEN-i))]
                    if (LEN-i)<tcn_num:
                        for _ in range(tcn_num-(LEN-i)):
                            randint=random.randint(i,LEN-1)
                            choose_num.append(randint)
                        choose_num.sort()

                    for k in choose_num:
                        item.append(person[k])
                    items.append(item)

                    i+=tcn_num

                if person_test:
                    itemss.append(items)
                    items=[]


        # if self.train:
        #     classnum=[0]*5
        #     classdiv=[[]]*5
        #     for item in items:
        #         whichone=int(min(4,item[0][-1]//0.2))
        #         classnum[whichone]+=1
        #         classdiv[whichone].append(item)

        #     MAXNNUM=max(classnum)
        #     for i in range(5):
        #         for _ in range(MAXNNUM-classnum[i]):
        #             items.append(classdiv[i][random.randint(0,classnum[i]-1)])

        self.items=items
        if person_test:
            self.items=itemss
        print(len(self.items))

    def load_img(self,file_path,hf=0.0,vf=0.0):
        img = Image.open(file_path)
        
        img=self.transform(img)
        if hf>0.5:
            img=tf.hflip(img)
        if vf>0.5:
            img=tf.vflip(img)
        return img

    def load_npy(self,file_path):
        # npy=np.load(file_path)
        # npy=npyStandard(npy)
        # npy=torch.from_numpy(npy)
        # npy=npy.to(torch.float32)
        npy=torch.tensor([0], dtype=torch.int)
        return npy

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idr):
        item=self.items[idr]
        imgs=[]
        npys=[]
        label=0.0
        sample = None
        if not self.is_person:
            img=self.load_img(item[0])
            npy=self.load_npy(item[1])
            sample = {'x1': img,'x2':npy,'y':item[-1]}
        else:
            if not self.person_test:
                hf,vf=0.0,0.0
                if self.train:
                    hf=random.random()
                    vf=random.random()
                for it in item:
                    imgs.append(self.load_img(it[0],hf,vf))
                    npys.append(self.load_npy(it[1]))
                    label=it[-1]
                imgs=torch.stack(imgs)
                npys=torch.stack(npys)
                sample = {'x1': imgs,'x2':npys,'y':label}
            else:
                imgss=[]
                npyss=[]
                for ite in item:
                    imgs=[]
                    npys=[]
                    for it in ite:
                        imgs.append(self.load_img(it[0]))
                        npys.append(self.load_npy(it[1]))
                        label=it[-1]
                    imgs=torch.stack(imgs)
                    npys=torch.stack(npys)
                    imgss.append(imgs)
                    npyss.append(npys)
                sample = {'x1': imgss,'x2':npyss,'y':label}
        return sample

class VoiceDataset(Dataset):
    def __init__(self,train,train_rio,paths,is_person,tcn_num=1,person_test=0):
        self.train=train
        self.is_person=is_person
        self.items=[]
        self.person_test=person_test

        for path in paths:
            self.items+=getVoiceSample(path[0],path[1],path[2])

        self.train_rio=int(len(self.items)*train_rio)

        if train:
            self.items=self.items[:self.train_rio]
            
        else:
            self.items=self.items[self.train_rio:]
            

        items=[]
        itemss=[]
        
        if not is_person:
            for person in self.items:
                for wav in person:
                    items.append(wav)
            self.items=items
        else:
            for person in self.items:
                i=0
                LEN=len(person)

                while i<LEN:
                    item=[]
                    choose_num=[x for x in range(i,i+min(tcn_num,LEN-i))]
                    if (LEN-i)<tcn_num:
                        for _ in range(tcn_num-(LEN-i)):
                            randint=random.randint(i,LEN-1)
                            choose_num.append(randint)
                        choose_num.sort()

                    for k in choose_num:
                        item.append(person[k])
                    items.append(item)

                    i+=tcn_num

                if person_test:
                    itemss.append(items)
                    items=[]

        self.items=items
        if person_test:
            self.items=itemss
        print(len(self.items))

    def load_npy(self,file_path):
        npy=np.load(file_path)
        npy=torch.from_numpy(npy)
        npy=torch.unsqueeze(npy, 0)
        npy=npy.to(torch.float32)
        return npy

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idr):
        item=self.items[idr]
        npys=[]
        label=0.0
        sample = None
        if not self.is_person:
            npy=self.load_npy(item[0])
            sample = {'x':npy,'xx':[0],'y':item[-1]}
        else:
            if not self.person_test:
                for it in item:
                    npys.append(self.load_npy(it[0]))
                    label=it[-1]
                
                npys=torch.stack(npys)
                sample = {'x':npys,'xx':[0],'y':label}

            else:
                npyss=[]
                for ite in item:
                    npys=[]
                    for it in ite:
                        npys.append(self.load_npy(it[0]))
                        label=it[-1]
                    npys=torch.stack(npys)
                    npyss.append(npys)
                sample = {'x': npyss,'xx':[0],'y':label}
        return sample

class FVDataset(Dataset):
    def __init__(self,train,train_rio,paths,tcn_num=1,person_test=0):
        self.train=train
        self.items=[]
        self.person_test=person_test
        self.transform=Compose([Resize([96,96]),ToTensor()])

        for path in paths:
            self.items+=getFaceSample(os.path.join(os.path.dirname(path[0]),'face'),path[1],path[2])

        fvitems=[]
        for i in range(len(self.items)):
            fvitem=[]
            for j in range(len(self.items[i])):
                splits=self.items[i][j][0].split('/')
                splits[0]='/'+splits[0]
                splits[8]='voice'
                splits[10]=splits[10].split('.')[0]+'.wav_fftnpy'
                splits[11]=splits[11].split('.')[0]+'.npy'
                voicepath=os.path.join(*splits)
                if os.path.exists(voicepath):
                    self.items[i][j].insert(1,voicepath)
                    fvitem.append(self.items[i][j])
            if len(fvitem)>0:
                fvitems.append(fvitem)
        
        self.items=fvitems


        self.train_rio=int(len(self.items)*train_rio)

        if train:
            self.items=self.items[:self.train_rio]
            
        else:
            self.items=self.items[self.train_rio:]
            

        items=[]
        itemss=[]
        
    
        for person in self.items:
            i=0
            LEN=len(person)

            while i<LEN:
                item=[]
                choose_num=[x for x in range(i,i+min(tcn_num,LEN-i))]
                if (LEN-i)<tcn_num:
                    for _ in range(tcn_num-(LEN-i)):
                        randint=random.randint(i,LEN-1)
                        choose_num.append(randint)
                    choose_num.sort()

                for k in choose_num:
                    item.append(person[k])
                items.append(item)

                i+=tcn_num

            if person_test:
                itemss.append(items)
                items=[]

        self.items=items
        if person_test:
            self.items=itemss
        print(len(self.items))

    def load_npy(self,file_path):
        npy=np.load(file_path)
        npy=torch.from_numpy(npy)
        npy=torch.unsqueeze(npy, 0)
        npy=npy.to(torch.float32)
        return npy

    def load_img(self,file_path,hf=0.0,vf=0.0):
        img = Image.open(file_path)
        
        img=self.transform(img)
        if hf>0.5:
            img=tf.hflip(img)
        if vf>0.5:
            img=tf.vflip(img)
        return img

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idr):
        item=self.items[idr]
        imgs=[]
        npys=[]
        label=0.0
        sample = None
        if not self.person_test:
            hf,vf=0.0,0.0
            if self.train:
                hf=random.random()
                vf=random.random()
            for it in item:
                imgs.append(self.load_img(it[0],hf,vf))
                npys.append(self.load_npy(it[1]))
                label=it[-1]
            imgs=torch.stack(imgs)
            npys=torch.stack(npys)
            sample = {'x1': imgs,'x2':npys,'y':label}
        else:
            imgss=[]
            npyss=[]
            for ite in item:
                imgs=[]
                npys=[]
                for it in ite:
                    imgs.append(self.load_img(it[0]))
                    npys.append(self.load_npy(it[1]))
                    label=it[-1]
                imgs=torch.stack(imgs)
                npys=torch.stack(npys)
                imgss.append(imgs)
                npyss.append(npys)
            sample = {'x1': imgss,'x2':npyss,'y':label}        
        return sample

def get_face_normal(paths):
    items=[]
    items2=[]
    transforms=Resize([96,96])
    for path in paths:
        items+=getFaceSample(path[0],path[1],path[2])
    for person in items:
        for img in person:
            items2.append(img)

    means=[0,0,0]
    stds=[0,0,0]
    for item in items2:
        img = Image.open(item[0])
        
        img=transforms(img)
        img=np.array(img)
        
        for i in range(3):
            pixels =img[:,:,i].ravel()
            pixels = pixels.astype(np.float32) / 255.
            means[i]+=np.mean(pixels)
            stds[i]+=np.std(pixels)
            
    means=np.array(means)
    stds=np.array(stds)
    means/=len(items2)
    stds/=len(items2)
    print(means,stds)
        
eye_mean=[[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]]]
mouth_mean=[[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]]]
def setMean(face_points,label):
    eye=face_points[:,0,5:-1].mean(-1)
    eye_mean[int(label//0.1)][0].append(float(eye.min()))
    eye_mean[int(label//0.1)][1].append(float(eye.mean()))
    eye_mean[int(label//0.1)][2].append(float(eye.max()))
    mouth=face_points[:,1,2:5].mean(-1)
    mouth_mean[int(label//0.1)][0].append(float(mouth.min()))
    mouth_mean[int(label//0.1)][1].append(float(mouth.mean()))
    mouth_mean[int(label//0.1)][2].append(float(mouth.max()))

class AllDataset(Dataset):
    def __init__(self,is_train,train_rio,paths,modal,is_time,pic_size):
        self.is_time=is_time
        self.modal=0 if modal=='face' else 1 if modal=='voice' else 0.5 if modal=='face_point' else 2
        self.all_items=[]
        self.fv_items=[]
        self.items=[]
        if is_train and not is_time:
            self.pic_transform = Compose([
                        Resize([pic_size,pic_size]),
                        # RandomHorizontalFlip(),
                        # RandomVerticalFlip(),
                        # ColorJitter(0.1,0.1,0.1,0.1),
                        ToTensor()])
        else:
            self.pic_transform = Compose([
                    Resize([pic_size,pic_size]),
                    ToTensor()])
        for path in paths:
            self.all_items+=getAllSample(*path)
        for sample in self.all_items:
            if len(sample)==3:
                self.fv_items.append(sample)
            else:
                self.items.append(sample)
        self.all_items=self.items
        self.items=[]
        self.train_rio=int(len(self.all_items)*train_rio)
        self.all_items=self.all_items[:self.train_rio] if is_train else self.all_items[self.train_rio:]
        if self.modal<2:
            if is_train:
                self.all_items+=self.fv_items
            if not is_time:
                houzhui='jpg' if self.modal==0 else 'npy'
                for item in self.all_items:
                    for img in sorted(os.listdir(item[floor(self.modal)]),key=lambda x:int(x.split('.')[0])):
                        if img.endswith(houzhui):
                            if self.modal==1 and not self.voice_open_mouth(os.path.join(item[floor(self.modal)],img),0.2):
                                continue
                            self.items.append([os.path.join(item[floor(self.modal)],img),item[-1]])
                self.all_items=self.items
        print(len(self.all_items))

    def __len__(self):
        return len(self.all_items)

    def load_img(self,file_path):
        img = Image.open(file_path)
        img = self.pic_transform(img)
        return img

    def load_npy(self,file_path):
        npy=np.load(file_path)
        npy=torch.from_numpy(npy)
        npy=torch.unsqueeze(npy, 0)
        npy=npy.to(torch.float32)/-100.0
        return npy

    def dis(self,p1,p2):
        ans=0
        for i in range(p1.shape[0]):
            ans+=(p1[i]-p2[i])**2
        return sqrt(ans)

    def dotproduct(self,p1,p2):
        ans=0
        for i in range(p1.shape[0]):
            ans+=p1[i]*p2[i]
        return ans

    def angle(self,p1,p2,p3):
        return acos(self.dotproduct(p1-p2,p1-p3)/(self.dis(p1,p2)*self.dis(p1,p3)))

    def face_rotate_angle(self,file_path):
        npy=np.load(file_path)
        angle=acos(self.dotproduct(npy[8][:2]-npy[27][:2],np.array([0,1]))/self.dis(npy[8][:2],npy[27][:2]))
        if (npy[8][:2]-npy[27][:2])[0]>0:
            angle=2.0*pi-angle
        return angle/pi*180

    def dis_point_line(self,p1,p2,p3):
        angle=self.angle(p1,p2,p3)
        ans=abs(self.dis(p1,p3)*sin(angle))/self.dis(p1,p2)
        return ans

    def load_face_point(self,file_path):
        npy=np.load(file_path)
        #eyes
        aus_eye=[
            self.dis(npy[44],npy[46])/self.dis(npy[42],npy[45]),
            self.dis(npy[43],npy[47])/self.dis(npy[42],npy[45]),
            self.dis(npy[38],npy[40])/self.dis(npy[36],npy[39]),
            self.dis(npy[37],npy[41])/self.dis(npy[36],npy[39])
        ]
        for i in range(17,22):
            aus_eye.append(self.dis_point_line(npy[36],npy[39],npy[i]))
        for i in range(22,27):
            aus_eye.append(self.dis_point_line(npy[42],npy[45],npy[i]))
        #nose
        aus_eye.append(1.0/(self.dis(npy[27],npy[30])/self.dis(npy[31],npy[35])))
        #mouth
        aus_mouth=[
            1.0/(self.dis(npy[60],npy[64])/self.dis(npy[31],npy[35])),
            1.0/(self.dis(npy[48],npy[54])/self.dis(npy[31],npy[35]))
        ]
        for i in range(61,64):
            aus_mouth.append(self.dis(npy[i],npy[61+67-i])/self.dis(npy[60],npy[64]))
        for i in range(49,54):
            aus_mouth.append(self.dis(npy[i],npy[49+59-i])/self.dis(npy[48],npy[54]))
        for i in range(49,54):
            aus_mouth.append(self.dis_point_line(npy[31],npy[35],npy[i]))
        return torch.Tensor([aus_eye,aus_mouth])

    def get_feature(self, arr):
        return (arr) / (np.max(arr) - np.min(arr))


    def load_seq(self,file_path):
        data = np.loadtxt(file_path, delimiter=";", dtype=np.double)
        seq=np.array([self.get_feature(data[:,1]), self.get_feature(data[:,3]), self.get_feature(data[:,4])])
        return seq 

    def voice_open_mouth(self,file_path,rio):
        splits=file_path.split('/')
        splits[0]='/'+splits[0]
        splits[-4]='face'
        splits[-2]=splits[-2][:-10]+'mp4'
        npypath=os.path.join(*splits)
        
        if not os.path.exists(npypath):
            return False
        npy=np.load(npypath)
        aus_mouth=[]
        for i in range(61,64):
            aus_mouth.append(self.dis(npy[i],npy[61+67-i])/self.dis(npy[60],npy[64]))
        for i in range(49,54):
            aus_mouth.append(self.dis(npy[i],npy[49+59-i])/self.dis(npy[48],npy[54]))
        if sum(aus_mouth)/len(aus_mouth)<rio:
            return False
        return True
    

    def __getitem__(self,idr):
        item=self.all_items[idr]
        imgs=[]
        npys=[]
        bios=[]
        face_points=[]
        label=item[-1]
        angle=0
        if self.is_time:
            for img in sorted(os.listdir(item[0]),key=lambda x:int(x.split('.')[0])):
                if img.endswith('jpg'):
                    img_one=self.load_img(os.path.join(item[0],img))
                    npypath=os.path.join(item[0],img)[:-3]+'npy'
                    angle=self.face_rotate_angle(npypath)
                    img_one=rotate(img_one,angle)
                    voice_path=os.path.join(item[1],img)[:-3]+'npy'
                    if os.path.exists(voice_path):
                        imgs.append(img_one)
                        npys.append(self.load_npy(voice_path))
                        face_points.append(self.load_face_point(npypath))
            # for npy in sorted(os.listdir(item[1]),key=lambda x:int(x.split('.')[0])):
            #     if npy.endswith('npy'):
            #         npys.append(self.load_npy(os.path.join(item[1],npy)))
            imgs=torch.stack(imgs)
            npys=torch.stack(npys)
            face_points=torch.stack(face_points)
        else:
            if self.modal==0:
                imgs=self.load_img(item[0])

                npypath=item[0][:-3]+'npy'
                face_points=np.load(npypath)
                angle=self.face_rotate_angle(npypath)
                imgs=rotate(imgs,angle)      
            elif self.modal==1:
                npys=self.load_npy(item[0])
            else:
                face_points=self.load_face_point(item[0])
        if len(item)==4:
            pass
            #bios=self.load_seq(item[2])
        # if self.is_time:
        #     setMean(face_points,label)
        return {'xs': [imgs,npys,bios,face_points,angle],'y':label}

def collate_fn(batch_datas):
    modals=[[],[],[],[]]
    labels = []
    for data in batch_datas:
        for i in range(4):
            modals[i].append(data['xs'][i])
        labels.append(data['y'])
    labels = torch.tensor(labels)
    return {'xs': modals,'y':labels}

def main():
    DATA_PATHS=[
    ["/hdd/sdd/lzq/DLMM_new/dataset/2022.1.29/pain2","/hdd/sdd/lzq/DLMM_new/dataset/2022.1.29/pain2/label.csv",1],
    ["/hdd/sdd/lzq/DLMM_new/dataset/2022.2.25/pain1","/hdd/sdd/lzq/DLMM_new/dataset/2022.2.25/pain1/label.csv",1],
    ["/hdd/sdd/lzq/DLMM_new/dataset/2022.2.25/pain2","/hdd/sdd/lzq/DLMM_new/dataset/2022.2.25/pain2/label.csv",1],
    ["/hdd/sdd/lzq/DLMM_new/dataset/2022.2.25/pain3","/hdd/sdd/lzq/DLMM_new/dataset/2022.2.25/pain3/label.csv",1],
    ["/hdd/sdd/lzq/DLMM_new/dataset/2022.2.25/pain4","/hdd/sdd/lzq/DLMM_new/dataset/2022.2.25/pain4/label.csv",2],
    ["/hdd/sdd/lzq/DLMM_new/dataset/2022.3.5/pain3","/hdd/sdd/lzq/DLMM_new/dataset/2022.3.5/label.csv",2],
    ["/hdd/sdd/lzq/DLMM_new/dataset/2022.3.5/pain4","/hdd/sdd/lzq/DLMM_new/dataset/2022.3.5/label.csv",2],
    #["/hdd/sdd/lzq/DLMM_new/dataset/2022.3.5/pain5","/hdd/sdd/lzq/DLMM_new/dataset/2022.3.5/label.csv",2],
    ["/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/pain1","/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/label.csv",2],
    ["/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/pain2","/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/label.csv",2],
    ["/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/pain3","/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/label.csv",2],
    ["/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/pain4","/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/label.csv",2],
    ["/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/pain5","/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/label.csv",2],
    ["/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/pain7","/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/label.csv",2],
    ["/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/pain8","/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/label.csv",2]
    ]
    train_dataset=AllDataset(0,0.0,DATA_PATHS,'face',1)
    train_dataloader = DataLoader(train_dataset, batch_size=1,shuffle=True)
    from tqdm import tqdm
    bar = tqdm(total=len(train_dataloader))
    for i,data in enumerate(train_dataloader):
        bar.update(1)
    bar.close()
    eye_mean_mean=[[],[],[],[],[],[],[],[],[],[]]
    for i in range(10):
        for e in eye_mean[i]:
            if len(e)>0:
                eye_mean_mean[i].append(sum(e)/len(e))
    mouth_mean_mean=[[],[],[],[],[],[],[],[],[],[]]
    for i in range(10):
        for e in mouth_mean[i]:
            if len(e)>0:
                mouth_mean_mean[i].append(sum(e)/len(e))
    print(eye_mean_mean)
    print(mouth_mean_mean)