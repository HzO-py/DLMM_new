# from models import VGG,Classifier
import torch
# import torch.nn as nn
# from models import VGG
# import os
# from PIL import Image
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
# from torchvision.transforms import ToTensor, Resize, RandomCrop,Compose,RandomHorizontalFlip
# from torchvision.transforms.functional import rotate
# import dlib
import cv2
# import face_alignment
# from skimage import io
# detector = dlib.get_frontal_face_detector()

# def load_img(file_path):
#     transform = Compose([Resize(64),RandomCrop(48),RandomHorizontalFlip(),ToTensor()])
#     img=np.array(Image.open(file_path).convert('L'))
#     img = img[:, :, np.newaxis]
#     img = np.concatenate((img, img, img), axis=2)
#     img = Image.fromarray(img)
#     img=transform(img)
#     return img

# def main(file_path):
#     net2 = VGG("VGG19")
#     checkpoint = torch.load(os.path.join('/hdd/lzq/facetrain/logs', 'face.t7'))
#     net2.load_state_dict(checkpoint['net2'])

#     net3 = Classifier(512,2)
#     net3.load_state_dict(checkpoint['net3'])

#     net2.eval()
#     net3.eval()

#     x=load_img(file_path)
#     x=x.unsqueeze(0)
#     x = net2(x)
#     outputs = net3(x)
#     _, predicted = torch.max(outputs.data, 1)
#     print(predicted)

# def face_points_detect(filename):
    
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False,device="cuda")

#     input = io.imread(filename)

#     preds = fa.get_landmarks(input)[0]
#     x1=int(np.min(preds[:,0]))
#     x2=int(np.max(preds[:,0]))
#     y1=int(np.min(preds[:,1]))
#     y2=int(np.max(preds[:,1]))
#     print(x1,x2,y1,y2)
#     io.imsave("../../test/face_img.jpg",input[y1:y2,x1:x2])
#     print(list(preds[:,0]))
#     print(list(preds[:,1]))
#     print(list(preds[:,2]))
#     # img=cv2.imread(filename)
#     # gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     # faces = detector(gray_img, 1)
#     # if len(faces)==0:
#     #     return

#     # u=0
#     # maxn=0
#     # for i in range(len(faces)):
#     #     (x1,x2,y1,y2)=(faces[i].left(),faces[i].right(),faces[i].top(),faces[i].bottom())
#     #     h=y2-y1
#     #     w=x2-x1
#     #     if h*w>maxn:
#     #         u=i
#     # (x1,x2,y1,y2)=(faces[u].left(),faces[u].right(),faces[u].top(),faces[u].bottom())
#     # face_img=img[y1:y2,x1:x2]
#     # cv2.imwrite("/hdd/sdd/lzq/DLMM_new/test/face_img.jpg",face_img)

# #face_points_detect("/hdd/sdd/lzq/DLMM_new/test/img.jpg")
# #main("/home/lzq/srp/DLMM_new/test/pain.png")
# # input = io.imread("/hdd/sdd/lzq/DLMM_new/test/img.jpg")
# # input2=cv2.imread("/hdd/sdd/lzq/DLMM_new/test/img.jpg")
# # input2=cv2.cvtColor(input2,cv2.COLOR_BGR2RGB)
# # print(input.shape)
# import scipy.io.wavfile as wav
# from python_speech_features import mfcc
# import time

# # starttime = time.time()
# # for i in range(29):
# #     (rate,sig) = wav.read('/hdd/sdd/lzq/DLMM_new/dataset/2022.2.25/pain2/voice/17/17-ZZC-02.wav_folder/'+str(i)+'.wav')

# #     x = mfcc(sig,rate)

# # endtime = time.time()
# # dtime = endtime - starttime
# # print(dtime)
# from math import *
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# transform = Compose([Resize([44,44]),ToTensor()])
# def load_img(file_path):
#     img = Image.open(file_path).convert('L')
#     img=np.array(img)
#     img = img[:, :, np.newaxis]
#     img = np.concatenate((img, img, img), axis=2)
#     img = Image.fromarray(img)
#     img = transform(img)
#     return img

# def dis(p1,p2):
#     ans=0
#     for i in range(p1.shape[0]):
#         ans+=(p1[i]-p2[i])**2
#     return sqrt(ans)

# def dotproduct(p1,p2):
#     ans=0
#     for i in range(p1.shape[0]):
#         ans+=p1[i]*p2[i]
#     return ans

# def load_face_point(file_path):
#     npy=np.load(file_path)
#     angle=acos(dotproduct(npy[8][:2]-npy[27][:2],np.array([0,1]))/dis(npy[8][:2],npy[27][:2]))
#     if (npy[8][:2]-npy[27][:2])[0]>0:
#         angle=2.0*pi-angle
#     return angle/pi*180

# emo=['angry','disgust','fear','happy','sad','surprise','neutral']
# modal=VGG('VGG19').cuda()
# modal.load_state_dict(torch.load('/hdd/sdd/lzq/DLMM_new/model/PrivateTest_model.t7')['net'])
# path='/hdd/sdd/lzq/DLMM_new/dataset/2022.3.5/pain5/face/404/404-SLY-04.mp4'
# for file in sorted(os.listdir(path),key=lambda x:int(x.split('.')[0])):
#     if file.endswith('jpg'):
#         filepath=os.path.join(path,file)
#         npypath=os.path.join(path,file.split('.')[0]+'.npy')
#         x=load_img(filepath)
#         angle=load_face_point(npypath)
#         x=rotate(x,angle)
#         x=x.unsqueeze(0).cuda()
#         out,fea=modal(x)
#         print(file,angle,emo[out[0].argmax()],out[0].max())
#         #print(file,np.load(filepath)[31:35,:2])

# import os
# import shutil
# paths=[
#     "/hdd/sdd/lzq/DLMM_new/dataset/2022.1.29/pain2",
#     "/hdd/sdd/lzq/DLMM_new/dataset/2022.2.25/pain1",
#     "/hdd/sdd/lzq/DLMM_new/dataset/2022.2.25/pain2",
#     "/hdd/sdd/lzq/DLMM_new/dataset/2022.2.25/pain3",
#     "/hdd/sdd/lzq/DLMM_new/dataset/2022.2.25/pain4",
#     "/hdd/sdd/lzq/DLMM_new/dataset/2022.3.5/pain3",
#     "/hdd/sdd/lzq/DLMM_new/dataset/2022.3.5/pain4",
#     #"/hdd/sdd/lzq/DLMM_new/dataset/2022.3.5/pain5",
#     "/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/pain1",
#     "/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/pain2",
#     "/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/pain3",
#     "/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/pain4",
#     "/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/pain5",
#     "/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/pain7",
#     "/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/pain8",
#   ]
# for path in paths:
#     face_path=os.path.join(path,'face')
#     shutil.rmtree(face_path)
#     print(face_path)
# logmelspec=np.load("/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/pain2/voice/140/140-QRX-04.wav_fftnpy/0.npy")
# print(logmelspec)
# plt.figure()
# librosa.display.specshow(logmelspec, sr=44100,x_axis='time',y_axis='mel')
# #plt.axis('off')
# plt.savefig('/hdd/sdd/lzq/DLMM_new/project/DLMM_new/test/voice.jpg')
# print(librosa.mel_to_hz(64))
a=torch.Tensor([[9,4,3],[1,2,3]])
b=torch.Tensor([[1,1,0],[0,1,1]])
c=torch.zeros((8,8))
for i in range(8):
    for j in range(8):
        if abs(i-j)<2:
            c[i][j]=1
print(c)