import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 7)

    def forward(self, x):
        out = self.features(x)
        fea = out.view(out.size(0), -1)
        #out = F.dropout(fea, p=0.5, training=self.training)
        #out = self.classifier(out)
        return fea

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class Prototype(nn.Module):#自定义类 继承nn.Module

    def __init__(self,inputNum,hiddenNum,outputNum):#初始化函数
        super(Prototype, self).__init__()#继承父类初始化函数

        self.fc1 = nn.Linear(inputNum, hiddenNum, bias = False)
        self.fc2 = nn.Linear(hiddenNum, outputNum, bias = False)

    def forward(self, x):
        out = self.fc1(x)
  
        fc_w1 = list(self.fc1.parameters())
        fc_w2 = list(self.fc2.parameters())

        return out,fc_w1,fc_w2

class Classifier(nn.Module):# 最终的分类器，用于输出预测概率

    def __init__(self,inputNum,outputNum,hiddenNum=6):#初始化函数
        super(Classifier, self).__init__()#继承父类初始化函数
        #self.fc1 = nn.Linear(inputNum, hiddenNum, bias = True)
        self.fc2 = nn.Linear(inputNum, outputNum, bias = True)

    def forward(self, x):
        # x = self.fc1(x)
        # x = F.selu(x)
        out = self.fc2(x)
        return out 

class Regressor(nn.Module):# 最终的分类器，用于输出预测概率

    def __init__(self,inputNum,hiddenNum):#初始化函数
        super(Regressor, self).__init__()#继承父类初始化函数
        self.fc1 = nn.Linear(inputNum, hiddenNum, bias = True)
        self.fc2 = nn.Linear(hiddenNum, hiddenNum, bias = True)
        self.fc3 = nn.Linear(hiddenNum, 1, bias = False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.selu(x)
        x = self.fc2(x)
        x = F.selu(x)
        out = self.fc3(x)
        return out 

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=28):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer5 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer6 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        
        #self.fc = nn.Linear(512, 28)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        #out = self.layer6(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        xx = out
        #out = self.fc(out)

        return xx


def ResNet18():

    return ResNet(ResidualBlock)

class cnn1d(nn.Module):#自定义类 继承nn.Module

    def __init__(self,outputNum):#初始化函数
        super(cnn1d, self).__init__()#继承父类初始化函数

        self.fc1 = nn.Linear(22336, 256, bias = True)

        self.fc2 = nn.Linear(256, outputNum, bias = True)   
   
        #self.fc3 = nn.Linear(256, 32, bias = True)
 
        #self.fc4 = nn.Linear(128, 32, bias = True)
        self.stride=2
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.max_pool1 = nn.MaxPool1d(kernel_size=3, stride=self.stride)
        
    def forward(self, x):
        # x = F.tanh(self.fc1(x))
        # x = F.tanh(self.fc2(x))
        # x = F.tanh(self.fc3(x))
        # out = F.tanh(self.fc4(x))
        #print(x.shape)
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = self.max_pool1(x)
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = self.max_pool1(x)
        #print(x.shape)
        x = F.relu(self.conv3(x))
        #print(x.shape)
        x = self.max_pool1(x)
        #print(x.shape)
        x = x.view(x.size(0),-1)
        #print(x.shape)
   
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #x = F.relu(self.fc4(x))
        out = x
        return out

