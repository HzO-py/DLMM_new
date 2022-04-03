from models import VGG,Classifier
import torch
import os
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor, Resize, RandomCrop,Compose,RandomHorizontalFlip
import dlib
import cv2
import face_alignment
from skimage import io
detector = dlib.get_frontal_face_detector()

def load_img(file_path):
    transform = Compose([Resize(64),RandomCrop(48),RandomHorizontalFlip(),ToTensor()])
    img=np.array(Image.open(file_path).convert('L'))
    img = img[:, :, np.newaxis]
    img = np.concatenate((img, img, img), axis=2)
    img = Image.fromarray(img)
    img=transform(img)
    return img

def main(file_path):
    net2 = VGG("VGG19")
    checkpoint = torch.load(os.path.join('/hdd/lzq/facetrain/logs', 'face.t7'))
    net2.load_state_dict(checkpoint['net2'])

    net3 = Classifier(512,2)
    net3.load_state_dict(checkpoint['net3'])

    net2.eval()
    net3.eval()

    x=load_img(file_path)
    x=x.unsqueeze(0)
    x = net2(x)
    outputs = net3(x)
    _, predicted = torch.max(outputs.data, 1)
    print(predicted)

def face_points_detect(filename):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False,device="cuda")

    input = io.imread(filename)

    preds = fa.get_landmarks(input)[0]
    x1=int(np.min(preds[:,0]))
    x2=int(np.max(preds[:,0]))
    y1=int(np.min(preds[:,1]))
    y2=int(np.max(preds[:,1]))
    print(x1,x2,y1,y2)
    io.imsave("../../test/face_img.jpg",input[y1:y2,x1:x2])
    print(list(preds[:,0]))
    print(list(preds[:,1]))
    print(list(preds[:,2]))
    # img=cv2.imread(filename)
    # gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # faces = detector(gray_img, 1)
    # if len(faces)==0:
    #     return

    # u=0
    # maxn=0
    # for i in range(len(faces)):
    #     (x1,x2,y1,y2)=(faces[i].left(),faces[i].right(),faces[i].top(),faces[i].bottom())
    #     h=y2-y1
    #     w=x2-x1
    #     if h*w>maxn:
    #         u=i
    # (x1,x2,y1,y2)=(faces[u].left(),faces[u].right(),faces[u].top(),faces[u].bottom())
    # face_img=img[y1:y2,x1:x2]
    # cv2.imwrite("/hdd/sdd/lzq/DLMM_new/test/face_img.jpg",face_img)

#face_points_detect("/hdd/sdd/lzq/DLMM_new/test/img.jpg")
#main("/home/lzq/srp/DLMM_new/test/pain.png")
# input = io.imread("/hdd/sdd/lzq/DLMM_new/test/img.jpg")
# input2=cv2.imread("/hdd/sdd/lzq/DLMM_new/test/img.jpg")
# input2=cv2.cvtColor(input2,cv2.COLOR_BGR2RGB)
# print(input.shape)
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import time

starttime = time.time()
for i in range(29):
    (rate,sig) = wav.read('/hdd/sdd/lzq/DLMM_new/dataset/2022.2.25/pain2/voice/17/17-ZZC-02.wav_folder/'+str(i)+'.wav')

    x = mfcc(sig,rate,nfft=1103)

endtime = time.time()
dtime = endtime - starttime
print(dtime/29)