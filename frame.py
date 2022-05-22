import cv2
import os
import face_alignment
from skimage import io
import numpy as np
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False,device="cuda")

def is_iou(x1,x2,y1,y2,x1_pre,x2_pre,y1_pre,y2_pre):
    xs=[x1_pre,x2_pre]
    ys=[y1_pre,y2_pre]
    for x in xs:
        for y in ys:
            if x>=x1 and x<=x2 and y>=y1 and y<=y2:
                return True
    xs=[x1,x2]
    ys=[y1,y2]
    for x in xs:
        for y in ys:
            if x>=x1_pre and x<=x2_pre and y>=y1_pre and y<=y2_pre:
                return True
    return False

def face_points_detect(img,x1_pre,x2_pre,y1_pre,y2_pre):
    
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_h=img.shape[0]
    img_w=img.shape[1]
    faces = fa.get_landmarks(img)
    if faces==None:
        return None,None,x1_pre,x2_pre,y1_pre,y2_pre

    x1,x2,y1,y2=0,0,0,0
    maxn=0
    pos=None
    for face in faces:
        xx1=int(np.min(face[:,0]))
        xx2=int(np.max(face[:,0]))
        yy1=int(np.min(face[:,1]))
        yy2=int(np.max(face[:,1]))
        w=xx2-xx1
        h=yy2-yy1
        if h*w>maxn:
            x1,x2,y1,y2=xx1,xx2,yy1,yy2
            maxn=h*w
            pos=face
            if w<h:
                cha=(h-w)//2
                x1=max(x1-cha,0)
                x2=min(x2+cha,img_w)
            else:
                cha=(w-h)//2
                y1=max(y1-cha,0)
                y2=min(y2+cha,img_h)

    if not ((x1_pre,x2_pre,y1_pre,y2_pre)==(0,0,0,0) or is_iou(x1,x2,y1,y2,x1_pre,x2_pre,y1_pre,y2_pre)):
        return None,None,x1_pre,x2_pre,y1_pre,y2_pre
    
    face_img=img[y1:y2,x1:x2]

    return face_img,pos,x1,x2,y1,y2

def mp42img(paths):
    for path in paths:
        dir_path=os.path.join(path,'video')
        save_path=os.path.join(path,'face')

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for i in sorted(os.listdir(dir_path),key=lambda x:int(x.split('.')[0])):
            dir_path_2=os.path.join(dir_path,i)
            for j in sorted(os.listdir(dir_path_2)):
                dir_path_3=os.path.join(dir_path_2,j)
                cap = cv2.VideoCapture(dir_path_3)
                fps = cap.get(cv2.CAP_PROP_FPS)

                x1,x2,y1,y2=0,0,0,0
                save_path_2=os.path.join(save_path,i)
                if not os.path.exists(save_path_2):
                    os.mkdir(save_path_2)
                save_path_2=os.path.join(save_path_2,j)
                if not os.path.exists(save_path_2):
                    os.mkdir(save_path_2)
                cnt=0
                act_cnt=0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    act_cnt+=1
                    if act_cnt%round(fps)!=0:
                        continue
                    img,pos,x1,x2,y1,y2=face_points_detect(frame,x1,x2,y1,y2)
                    if img is None or img.shape[0]*img.shape[1]*img.shape[2]==0:
                        continue
                    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(save_path_2,str(cnt))+".jpg",img)
                    np.save(os.path.join(save_path_2,str(cnt)+".npy"),pos)
                    cnt+=1
                print(save_path_2)
                cap.release() 
     
mp42img([
    # "/hdd/sdd/lzq/DLMM_new/dataset/2022.1.29/pain2",
    # "/hdd/sdd/lzq/DLMM_new/dataset/2022.2.25/pain1",
    # "/hdd/sdd/lzq/DLMM_new/dataset/2022.2.25/pain2",
    # "/hdd/sdd/lzq/DLMM_new/dataset/2022.2.25/pain3",
    # "/hdd/sdd/lzq/DLMM_new/dataset/2022.2.25/pain4",
    # "/hdd/sdd/lzq/DLMM_new/dataset/2022.3.5/pain3",
    # "/hdd/sdd/lzq/DLMM_new/dataset/2022.3.5/pain4",
    # "/hdd/sdd/lzq/DLMM_new/dataset/2022.3.5/pain5",
    # "/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/pain1",
    # "/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/pain2",
    # "/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/pain3",
    # "/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/pain4",
    "/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/pain5",
  ])
