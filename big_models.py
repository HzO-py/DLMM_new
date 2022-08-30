from torch.nn.utils import weight_norm
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from models import BILSTM, Prototype,Classifier,ResNet18,cnn1d,VGG,Regressor,TemporalConvNet

class FaceModel(nn.Module):
    def __init__(self):
        super(FaceModel, self).__init__()
        self.extractor=VGG("VGG19")

    def forward(self, x):