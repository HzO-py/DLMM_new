import os
from scipy.io import wavfile
from scipy.io.wavfile import write
import os
import numpy as np
import moviepy.editor as mp
from tqdm import tqdm
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import librosa

def keepVoice(path,fileName):
    index = 0
    file_rate, file = wavfile.read(os.path.join(path, fileName))
    file=file[:,0]
    start = 0
    end = len(file)
    segmentLen = 100000
    checkLen=100
    myDB = -1
    SegmentCnt = (end-start) // segmentLen

    output_path=os.path.join(path, fileName+'_folder')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    print(output_path)


    for i in range(SegmentCnt):
        energy = 0.0
        for j in range(0,segmentLen,checkLen):
            res = 0.0
            for k in range(checkLen):
                res +=abs(file[i * segmentLen + j+k])
            res/=checkLen
            
            if res.any()>myDB:
                energy = 1
                break
        
        if energy > 0:

            newFile = np.concatenate([file[i * segmentLen:(i+1)*segmentLen]])
            
            newName = os.path.join(output_path, str(index) + ".wav")
            index += 1
            write(newName, file_rate, newFile)
        

def wav2npy(path):
    save_path=path+'_fftnpy'
    print(save_path)

    y, sr = librosa.load(path, sr=None)
    for i in range(len(y)//sr):
        y_sub=y[i*sr:(i+1)*sr]
        melspec = librosa.feature.melspectrogram(y_sub, sr, n_fft=2048, hop_length=512, n_mels=128)
        logmelspec = librosa.power_to_db(melspec)
        np.save(os.path.join(save_path,str(i)+'.npy'),logmelspec)
    

def listwav(paths):
    for path in paths:
        dir_path=os.path.join(path,'voice')
        for i in sorted(os.listdir(dir_path),key=lambda x:int(x.split('.')[0])):
            dir_path_2=os.path.join(dir_path,i)
            for j in sorted(os.listdir(dir_path_2)):
                if j.endswith('.wav'):
                    dir_path_3=os.path.join(dir_path_2,j+'_fftnpy')
                    if not os.path.exists(dir_path_3):
                        os.mkdir(dir_path_3)
                        wav2npy(os.path.join(dir_path_2,j))

                            
                
def mp42wav(paths):
    for path in paths:
        dir_path=os.path.join(path,'video')
        save_path=os.path.join(path,'voice')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for i in sorted(os.listdir(dir_path),key=lambda x:int(x.split('.')[0])):
            dir_path_2=os.path.join(dir_path,i)
            for j in sorted(os.listdir(dir_path_2)):
                dir_path_3=os.path.join(dir_path_2,j)
                save_path_2=os.path.join(save_path,i)
                if not os.path.exists(save_path_2):
                    os.mkdir(save_path_2)
                save_path_3=os.path.join(save_path_2,j.split('.')[0]+'.wav')
                
                my_clip = mp.VideoFileClip(dir_path_3)
                my_clip.audio.write_audiofile(save_path_3)
                

listwav([
    "/hdd/sdd/lzq/DLMM_new/dataset/2022.1.29/pain2",
    "/hdd/sdd/lzq/DLMM_new/dataset/2022.2.25/pain1",
    "/hdd/sdd/lzq/DLMM_new/dataset/2022.2.25/pain2",
    "/hdd/sdd/lzq/DLMM_new/dataset/2022.2.25/pain3",
    "/hdd/sdd/lzq/DLMM_new/dataset/2022.2.25/pain4",
    "/hdd/sdd/lzq/DLMM_new/dataset/2022.3.5/pain3",
    "/hdd/sdd/lzq/DLMM_new/dataset/2022.3.5/pain4",
    "/hdd/sdd/lzq/DLMM_new/dataset/2022.3.5/pain5",
    "/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/pain1",
    "/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/pain2",
    "/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/pain3",
    "/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/pain4",
  ])