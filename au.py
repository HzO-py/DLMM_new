import cv2
import os

import dlib
import numpy as np
from PIL import Image

from torchvision import utils as vutils
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import ToTensor, Scale, Compose
import torch
from torch.autograd import Variable

from models_multiview import FrontaliseModelMasks_wider
#获取一张图片的宽高作为视频的宽高
model = FrontaliseModelMasks_wider(inner_nc=256, \
            num_output_channels=3, num_masks=0, num_additional_ids=32)
old_model_name = '/home/lzq/srp/DLMM_new/tcae/tcae_epoch_1000.pth'
model.load_state_dict(torch.load(old_model_name)['state_dict'])
model.eval()

transforms=Compose([Scale((256,256)),ToTensor()])

def img2mp4():
    dir_path='/home/lzq/srp/DLMM_new/openpose/examples/media/images'
    x1,x2,y1,y2=0,0,0,0
    for i in sorted(os.listdir(dir_path),key=lambda x:int(x.split('.')[0])):
        file_name = dir_path + '/'+i
        img=cv2.imread(file_name)
        img,x1,x2,y1,y2=face_points_detect(img,x1,x2,y1,y2)
        if img is None:
            #print(i)
            continue
        cv2.imwrite('/home/lzq/srp/DLMM_new/openpose/examples/media/face_images/'+i,img)

    # image=cv2.imread('/home/lzq/srp/DLMM_new/openpose/examples/media/output_images/5.jpg')

    # image_info=image.shape
    # height=image_info[0]
    # width=image_info[1]

    # fps=30
    # fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    # video = cv2.VideoWriter('s.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height)) #创建视频流对象-格式一

    # dir_path='/home/lzq/srp/DLMM_new/openpose/examples/media/output_images'
    # for i in sorted(os.listdir(dir_path),key=lambda x:int(x.split('.')[0])):
    #     file_name = dir_path + '/'+i
    #     image=cv2.imread(file_name)
    #     video.write(image)  # 向视频文件写入一帧--只有图像，没有声音

def load_img(file_path):
    img = Image.open(file_path).convert('RGB')
    img = transforms(img)
    img = img.unsqueeze(0)
    return img

def get_au(input_image,target_image):
    global model
    model = model.cuda()

    target_image = Variable(target_image).cuda()
    offset = 1
    input_image = Variable(input_image).cuda()

    target_pose_f, target_exp_f = model.encoder(target_image)
    input_pose_f, input_exp_f = model.encoder(input_image)

    #pose_vector = torch.cat((input_pose_f, target_pose_f), 1)
    exp_vector = torch.cat((input_exp_f, target_exp_f), 1)

    all_exp_flow_samplers = model.expression_decoder(exp_vector) 
    exp_flow_samplers = all_exp_flow_samplers[:,0:2,:,:]
    exp_mask = all_exp_flow_samplers[:,-1,:,:].unsqueeze(1)
    
    exp_grid = np.linspace(-1,1, exp_flow_samplers.size(2))
    exp_grid = np.meshgrid(exp_grid, exp_grid)
    exp_grid = np.stack(exp_grid, 2)  # w x h x 2
    exp_grid = torch.Tensor(exp_grid).unsqueeze(0).repeat(target_exp_f.size(0), 1,1,1).cuda()
    exp_grid = Variable(exp_grid, requires_grad=False)
    exp_samplers = (exp_flow_samplers.permute(0,2,3,1) + exp_grid).clamp(min=-1,max=1)
    exp_image = F.grid_sample(input_image.detach(), exp_samplers)

    print(exp_vector.size(),exp_flow_samplers.size())

    exp_image=exp_image.squeeze(0)
    vutils.save_image(exp_image, "/home/lzq/srp/DLMM_new/tcae/exp_img.jpg")
    return exp_flow_samplers.squeeze(0).permute(1,2,0).cpu().detach().numpy()

def show_au(file_path,exp_flow_samplers):
    img=cv2.imread(file_path)
    img=cv2.resize(img,(256,256))
    grid_maxn=np.array([[[[0.]*2]*16]*16][0])
    for i in range(256):
        for j in range(256):      
            len=np.linalg.norm(exp_flow_samplers[i][j])
            if len>np.linalg.norm(grid_maxn[i//16][j//16]):
                grid_maxn[i//16][j//16]=exp_flow_samplers[i][j]
    for i in range(16):
        for j in range(16):
            rio=100
            len=np.linalg.norm(grid_maxn[i][j])
            if len>10:
                rio=10/len
            dx=int(grid_maxn[i][j][0]*rio)
            dy=int(grid_maxn[i][j][1]*rio)
            print(dx,dy)
            img=cv2.arrowedLine(img, (i*16+8,j*16+8), (i*16+8+dx,j*16+8+dy), (0,255,0),1,1,0,0.3)
    cv2.imwrite("/home/lzq/srp/DLMM_new/tcae/flow_img.jpg",img)
    

input_image=load_img("/home/lzq/srp/DLMM_new/openpose/examples/media/face_images/9.jpg")
target_image=load_img("/home/lzq/srp/DLMM_new/openpose/examples/media/face_images/11.jpg")
exp_flow_samplers=get_au(input_image,target_image)
show_au("/home/lzq/srp/DLMM_new/openpose/examples/media/face_images/9.jpg",exp_flow_samplers)