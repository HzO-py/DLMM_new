import argparse
from tkinter.messagebox import NO
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import *
from torchvision.transforms.functional import rotate
from math import *
from models import *
import os
import cv2
from utils import getCfg
from train_module import *
import librosa
import librosa.display
from torchvision.transforms.functional import rotate

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
CHECKPOINT_NAME=cfg["CHECKPOINT_NAME"]

TRAIN_RIO=cfg["TRAIN_RIO"]
DATA_PATHS=cfg["DATA_PATHS"]
PIC_SIZE=cfg["PIC_SIZE"]
IS_POINT=cfg["IS_POINT"]

class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, extractor,pic_size,classifier,checkpoint):
        self.extractor = extractor.eval()
        self.classifier = classifier.eval()
        self.extractor.cuda()
        self.classifier.cuda()
        self.extractor.load_state_dict(checkpoint['extractor'])
        self.classifier.load_state_dict(checkpoint['classifier'])
        self.pic_transform = Compose([
                    Resize([pic_size,pic_size]),
                    ToTensor()])

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

    def face_rotate_angle(self,file_path):
        npy=np.load(file_path)
        angle=acos(self.dotproduct(npy[8][:2]-npy[27][:2],np.array([0,1]))/self.dis(npy[8][:2],npy[27][:2]))
        if (npy[8][:2]-npy[27][:2])[0]>0:
            angle=2.0*pi-angle
        return angle/pi*180

    def load_img(self,file_path):
        img = Image.open(file_path)
        img = self.pic_transform(img)
        npypath=file_path[:-3]+'npy'
        angle=self.face_rotate_angle(npypath)
        img=rotate(img,angle)
        return img

    def get_cam_weights(self,grads,activations):
        grads_power_2 = grads**2
        grads_power_3 = grads_power_2 * grads
        # Equation 19 in https://arxiv.org/abs/1710.11063
        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 +
                               sum_activations[:, :, None, None] * grads_power_3 + eps)
        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0) * aij
        weights = np.sum(weights, axis=(2, 3))
        return weights

    def generate_cam(self, input_image,label=None):
        #1.1 前向传播，计算出目标类的最终输出值model_output，以及目标层的特征图的输出conv_output
        input_image=self.load_img(input_image).unsqueeze(0)
        _,fea,res = self.extractor(input_image)
        outputs,_=self.classifier(fea)
        if label is None:
            label=torch.argmax(outputs.data,1)
            print(label)
        one_hot_output = torch.FloatTensor(1, outputs.size()[-1]).zero_()
        one_hot_output[0][label] = 1
        # 步骤1.2 反向传播， 获取目标类相对于目标层各特征图的梯度
        target = res.data.numpy()
        # 步骤1.2.1 清零梯度：model.zero_grad()
        self.extractor.zero_grad()
        self.classifier.zero_grad()
        # 步骤1.2.2 计算反向传播
        outputs.backward(gradient=one_hot_output,retain_graph=True)
        # 步骤1.2.3 获取目标层各特征图的梯度
        guided_gradients = self.extractor.fea_grad.data.numpy()

        # 步骤2.1 对每张梯度图求均值，作为与其对应的特征图的权重
        weights=self.get_cam_weights(guided_gradients,target)[0]
        target=target[0]
        #weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # 初始化热力图
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # 步骤2.2 计算各特征图的加权值
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        #步骤2.3 对热力图进行后处理，即将结果变换到0~255
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        # cam=cv2.cvtColor(np.array(cam), cv2.COLOR_RGB2BGR)
        # cam=cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        cam_img=torch.tensor(cam)*input_image
        cam_img=cv2.cvtColor(cam_img.squeeze(0).mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy(),cv2.COLOR_RGB2BGR)
        #cam_img = 0.3*cam+0.7*img
        cv2.imwrite('/hdd/sdd/lzq/DLMM_new/project/DLMM_new/test/gradcam++_jiaquan.jpg', cam_img)
        return cam_img

    def load_npy(self,file_path):
        npy=np.load(file_path)
        npy=torch.from_numpy(npy)
        npy=torch.unsqueeze(npy, 0)
        npy=npy.to(torch.float32)/-100.0
        return npy

    def generate_cam_voice(self, input_image,label=None):
        #1.1 前向传播，计算出目标类的最终输出值model_output，以及目标层的特征图的输出conv_output
        logmelspec=np.load(input_image)
        plt.figure()
        librosa.display.specshow(logmelspec, sr=44100)
        plt.axis('off')
        plt.savefig('/hdd/sdd/lzq/DLMM_new/project/DLMM_new/test/voice.jpg',bbox_inches='tight',pad_inches=0.0)
        img=cv2.imread('/hdd/sdd/lzq/DLMM_new/project/DLMM_new/test/voice.jpg')

        input_image=self.load_npy(input_image).unsqueeze(0)
        _,fea,res = self.extractor(input_image)
        outputs,_=self.classifier(fea)
        if label is None:
            label=torch.argmax(outputs.data,1)
            print(label)
        one_hot_output = torch.FloatTensor(1, outputs.size()[-1]).zero_()
        one_hot_output[0][label] = 1
        # 步骤1.2 反向传播， 获取目标类相对于目标层各特征图的梯度
        target = res.data.numpy()
        # 步骤1.2.1 清零梯度：model.zero_grad()
        self.extractor.zero_grad()
        self.classifier.zero_grad()
        # 步骤1.2.2 计算反向传播
        outputs.backward(gradient=one_hot_output,retain_graph=True)
        # 步骤1.2.3 获取目标层各特征图的梯度
        guided_gradients = self.extractor.fea_grad.data.numpy()

        # 步骤2.1 对每张梯度图求均值，作为与其对应的特征图的权重
        weights=self.get_cam_weights(guided_gradients,target)[0]
        target=target[0]
        #weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # 初始化热力图
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # 步骤2.2 计算各特征图的加权值
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        #步骤2.3 对热力图进行后处理，即将结果变换到0~255
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((img.shape[1],
                       img.shape[0]), Image.ANTIALIAS))
        cam=cv2.cvtColor(np.array(cam), cv2.COLOR_RGB2BGR)
        cam=cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        # cam_img=torch.tensor(cam)*input_image
        # cam_img=cv2.cvtColor(cam_img.squeeze(0).mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy(),cv2.COLOR_RGB2BGR)
        cam_img = 0.3*cam+0.7*img
        cv2.imwrite('/hdd/sdd/lzq/DLMM_new/project/DLMM_new/test/voice_cam.jpg', cam_img)
        return cam_img

    def face_points_recover(self,face):
        face=face.numpy()
        xx1=int(np.min(face[:,0]))
        xx2=int(np.max(face[:,0]))
        yy1=int(np.min(face[:,1]))
        yy2=int(np.max(face[:,1]))
        w=xx2-xx1
        h=yy2-yy1
        x1,x2,y1,y2=xx1,xx2,yy1,yy2
        if w<h:
            cha=(h-w)//2
            x1=max(x1-cha,0)
            r=PIC_SIZE/h
        else:
            cha=(w-h)//2
            y1=max(y1-cha,0)
            r=PIC_SIZE/w
        for f in face:
            f[0]-=x1
            f[0]*=r
            f[0]=max(0,min(f[0],PIC_SIZE-1))
            f[1]-=y1
            f[1]*=r
            f[1]=max(0,min(f[1],PIC_SIZE-1))
        return face

    def all_generate_cam(self, input_image,label,face_points,angle):
        #1.1 前向传播，计算出目标类的最终输出值model_output，以及目标层的特征图的输出conv_output
        #input_image=self.load_img(input_image).unsqueeze(0)
        _,fea,res = self.extractor(input_image)
        outputs,_=self.classifier(fea)
        one_hot_output = torch.FloatTensor(1, outputs.size()[-1]).zero_().cuda()
        one_hot_output[0][label] = 1
        # 步骤1.2 反向传播， 获取目标类相对于目标层各特征图的梯度
        target = res.data.cpu().numpy()
        # 步骤1.2.1 清零梯度：model.zero_grad()
        self.extractor.zero_grad()
        self.classifier.zero_grad()
        # 步骤1.2.2 计算反向传播
        outputs.backward(gradient=one_hot_output,retain_graph=True)
        # 步骤1.2.3 获取目标层各特征图的梯度
        guided_gradients = self.extractor.fea_grad.data.cpu().numpy()

        # 步骤2.1 对每张梯度图求均值，作为与其对应的特征图的权重
        weights=self.get_cam_weights(guided_gradients,target)[0]
        target=target[0]
        #weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # 初始化热力图
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # 步骤2.2 计算各特征图的加权值
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        #步骤2.3 对热力图进行后处理，即将结果变换到0~255
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        # 
        cam=rotate(torch.tensor(cam).unsqueeze(0),360-float(angle)).squeeze(0)
        face_points=self.face_points_recover(face_points)
        #print(face_points)
        ans=np.ones(face_points.shape[0],dtype=np.float32)
        for i in range(face_points.shape[0]):
            ans[i]=cam[int(face_points[i][0])][int(face_points[i][1])]

        
        # cam=cv2.cvtColor(np.array(np.uint8(cam*255)), cv2.COLOR_RGB2BGR)
        # cam=cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        # img=rotate(input_image,360-float(angle)).squeeze(0).cpu()
        # img=cv2.cvtColor(img.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy(),cv2.COLOR_RGB2BGR)
        # cam_img = 0.3*cam+0.7*img
        # for p in face_points:
        #     cv2.circle(cam_img,(int(p[0]),int(p[1])),1,(0,255,0),-1)       
        # cv2.imwrite('/hdd/sdd/lzq/DLMM_new/project/DLMM_new/test/dian.jpg', cam_img)

        return ans

    def all_generate_cam_voice(self, input_image,label):
        #1.1 前向传播，计算出目标类的最终输出值model_output，以及目标层的特征图的输出conv_output
        #input_image=self.load_img(input_image).unsqueeze(0)
        logmelspec=Tensor.numpy(-100.0*input_image.squeeze(0).squeeze(0).cpu())
        plt.clf()
        librosa.display.specshow(logmelspec, sr=44100)
        plt.axis('off')
        plt.savefig('/hdd/sdd/lzq/DLMM_new/project/DLMM_new/test/voice.jpg',bbox_inches='tight',pad_inches=0.0)
        img=cv2.imread('/hdd/sdd/lzq/DLMM_new/project/DLMM_new/test/voice.jpg')

        _,fea,res = self.extractor(input_image)
        outputs,_=self.classifier(fea)
        one_hot_output = torch.FloatTensor(1, outputs.size()[-1]).zero_().cuda()
        one_hot_output[0][label] = 1
        # 步骤1.2 反向传播， 获取目标类相对于目标层各特征图的梯度
        target = res.data.cpu().numpy()
        # 步骤1.2.1 清零梯度：model.zero_grad()
        self.extractor.zero_grad()
        self.classifier.zero_grad()
        # 步骤1.2.2 计算反向传播
        outputs.backward(gradient=one_hot_output,retain_graph=True)
        # 步骤1.2.3 获取目标层各特征图的梯度
        guided_gradients = self.extractor.fea_grad.data.cpu().numpy()

        # 步骤2.1 对每张梯度图求均值，作为与其对应的特征图的权重
        weights=self.get_cam_weights(guided_gradients,target)[0]
        target=target[0]
        #weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # 初始化热力图
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # 步骤2.2 计算各特征图的加权值
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        #步骤2.3 对热力图进行后处理，即将结果变换到0~255
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((img.shape[1],
                       img.shape[0]), Image.ANTIALIAS))/255
        # cam=cv2.cvtColor(np.array(cam), cv2.COLOR_RGB2BGR)
        # cam=cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        # cam_img=torch.tensor(cam)*input_image
        # cam_img=cv2.cvtColor(cam_img.squeeze(0).mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy(),cv2.COLOR_RGB2BGR)
        # cam_img = 0.3*cam+0.7*img
        # cv2.imwrite('/hdd/sdd/lzq/DLMM_new/project/DLMM_new/test/voice_cam.jpg', cam_img)

        ans=np.mean(cam,axis=1)
        return ans

class allGradcam():
    def __init__(self, gradcam,dataset):
        self.gradcam=gradcam
        self.dataset=dataset
        self.points_weights=np.zeros(369,dtype=np.float32)

    def __call__(self):
        bar = tqdm(total=len(self.dataset.train_dataloader))
        cnt=0
        for data in self.dataset.train_dataloader:
            xs,y=data.values()
            x=xs[0]
            angle=xs[-1]
            face_points=xs[-2]
            y = y.to(torch.float32)
            y[y<=0.2] = 0
            y[y>0.2] = 1
            y=y.to(torch.long)
            x, y = x.cuda(), y.cuda()
            for i in range(x.size()[0]):
                if y[i]==1:
                    cnt+=1
                    points_weight=self.gradcam.all_generate_cam(x[i].unsqueeze(0),y[i].unsqueeze(0),face_points[i],angle[i])
                    self.points_weights+=points_weight
            bar.update(1)
        bar.close()
        self.points_weights/=cnt
        print(self.points_weights)

    def voice(self):
        bar = tqdm(total=len(self.dataset.train_dataloader))
        cnt=0
        for data in self.dataset.train_dataloader:
            xs,y=data.values()
            x=xs[1]
            y = y.to(torch.float32)
            y[y<=0.2] = 0
            y[y>0.2] = 1
            y=y.to(torch.long)
            x, y = x.cuda(), y.cuda()
            for i in range(x.size()[0]):
                if y[i]==1:
                    cnt+=1
                    points_weight=self.gradcam.all_generate_cam_voice(x[i].unsqueeze(0),y[i].unsqueeze(0))
                    self.points_weights+=points_weight
            bar.update(1)
        bar.close()
        self.points_weights/=cnt
        print(self.points_weights)

def one_GradCam(path):
    resgrad=GradCam(Resnet_regressor('voice',is_gradcam=True),PIC_SIZE,Classifier(EXTRACT_NUM,HIDDEN_NUM,CLASS_NUM),torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME)))
    cam=resgrad.generate_cam_voice(path)

#one_GradCam('/hdd/sdd/lzq/DLMM_new/dataset/2022.1.29/pain2/voice/2/02-CJR-02.wav_fftnpy/24.npy')
plt.figure()
all=allGradcam(GradCam(Resnet_regressor('voice',is_gradcam=True),PIC_SIZE,Classifier(EXTRACT_NUM,HIDDEN_NUM,CLASS_NUM),torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME))),DataSet(BATCH_SIZE,TRAIN_RIO,DATA_PATHS,'voice',is_time=False,collate_fn=None,pic_size=PIC_SIZE))
all.voice()