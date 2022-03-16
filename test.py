from models import VGG,Classifier
import torch
import os
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor, Resize, RandomCrop,Compose,RandomHorizontalFlip

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

main("/home/lzq/srp/DLMM_new/test/pain.png")