import cv2
import os
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/lzq/srp/facial/facial_emotion/FER2013_VGG19/shape_predictor_68_face_landmarks.dat")

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
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = detector(gray_img, 1)
    
    if len(faces)==0:
        return None,None,x1_pre,x2_pre,y1_pre,y2_pre

    u=0
    maxn=0
    for i in range(len(faces)):
        (x1,x2,y1,y2)=(faces[i].left(),faces[i].right(),faces[i].top(),faces[i].bottom())
        h=y2-y1
        w=x2-x1
        if h*w>maxn:
            u=i
   # face_img=img[y1:y2,x1:x2]
    (x1,x2,y1,y2)=(faces[u].left(),faces[u].right(),faces[u].top(),faces[u].bottom())
    if not ((x1_pre,x2_pre,y1_pre,y2_pre)==(0,0,0,0) or is_iou(x1,x2,y1,y2,x1_pre,x2_pre,y1_pre,y2_pre)):
        return None,None,x1_pre,x2_pre,y1_pre,y2_pre
        
    face_img=img[y1:y2,x1:x2]

    face_dect=predictor(gray_img, faces[u])
    pos=[]
    for i in range(68):
        pos.append([face_dect.part(i).x,face_dect.part(i).y])
        
    #print(img)
    return face_img,pos,x1,x2,y1,y2

def mp42img():
    dir_path="/hdd/lzq/data_2022.1.29/pain2/dataset"
    save_path="/hdd/lzq/data_2022.1.29/pain2/face"
    for i in sorted(os.listdir(dir_path),key=lambda x:int(x.split('.')[0])):
        # if int(i)<35:
        #     continue
        print(i)
        dir_path_2=os.path.join(dir_path,i)
        for j in sorted(os.listdir(dir_path_2)):
            dir_path_3=os.path.join(dir_path_2,j)
            cap = cv2.VideoCapture(dir_path_3)
            x1,x2,y1,y2=0,0,0,0
            save_path_2=os.path.join(save_path,i)
            if not os.path.exists(save_path_2):
                os.mkdir(save_path_2)
            save_path_2=os.path.join(save_path_2,j)
            if not os.path.exists(save_path_2):
                os.mkdir(save_path_2)
            cnt=0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                img,pos,x1,x2,y1,y2=face_points_detect(frame,x1,x2,y1,y2)
                if img is None or img.shape[0]*img.shape[1]*img.shape[2]==0:
                    continue
                cv2.imwrite(os.path.join(save_path_2,str(cnt))+".jpg",img)
                with open(os.path.join(save_path_2,str(cnt)+".txt"),mode='w') as f:
                    f.write(str(pos))
                cnt+=1
            print(save_path_2)
            cap.release() 
     
mp42img()
