""" 
Copyright 2023 AIFUTURE LLC.
Code for MMAC23 challenge from AIFUTURE Lab.
"""
import os
import sys
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)

import cv2
import torch
from torch import nn
import torchvision.models as models
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from torchvision.models import resnet50 as net 



weights = 'DEFAULT'

norm_mean= (0.485, 0.456, 0.406) # R,G,B
norm_std= (0.229, 0.224, 0.225)

size = (720, 720) # mae_vit规定输入, dino_vit没有规定

def train_transforms():
    return transforms.Compose([    
        transforms.Resize(size),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(512, padding=64),
        # transforms.RandomRotation(degrees=(0,180)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std) 
        ])

def val_transforms():
    return transforms.Compose([   
        transforms.Resize(size), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=norm_mean, std=norm_std)
        ])

def approximate_value(value):
    clamped_value = torch.clamp(value, min=-10, max=4)  # 将值限制在0-4范围内
    approximated_value = torch.round(clamped_value * 8) / 8  # 就近间隔0.125
    return approximated_value

class RegNet(nn.Module):
    def __init__(self, pretrain=False):
        super(RegNet, self).__init__()
        if pretrain:
            try:
                self.net = net(weights=weights)
            except:
                print("未加载 torchvision预训练模型")
                self.net = net()
        else:
            self.net = net()
        
        # 换掉最后一个全连接层为当前任务所需的分类层/回归层
        fc_name = list(net().state_dict().keys())[-1].split('.')[0]
        fc = getattr(self.net, fc_name)
        if fc_name == 'norm' and self.net.__class__.__name__ == 'VisionTransformer': # dino_vit
            feature_dim = self.net.norm.weight.shape[0]
        elif fc.__class__.__name__ == 'Sequential':
            assert fc[-1].__class__.__name__ == 'Linear'
            feature_dim = fc[-1].in_features
            fc[-1] = nn.Sequential()
        elif fc.__class__.__name__ == 'Linear':
            feature_dim = fc.in_features
            setattr(self.net, fc_name, 
                    nn.Sequential())
        else:
            raise ValueError("模型head比较特别")
        self.reg_head = nn.Sequential(nn.Linear(feature_dim, 128), nn.ReLU(), nn.Dropout(p=0.2), 
                                      nn.Linear(128, 32), nn.ReLU(), nn.Dropout(p=0.2), 
                                      nn.Linear(32, 1))
        # self.reg_head = nn.Linear(feature_dim, out_features=1)

        self.head = self.reg_head
    def forward(self, x):
        features = self.net(x)

        return self.head(features)
    

class model:
    def __init__(self):
        self.checkpoint = "model.pt"
        # The model is evaluated using CPU, please do not change to GPU to avoid error reporting.
        self.device = torch.device("cpu")

    def load(self, dir_path):
        """
        load the model and weights.
        dir_path is a string for internal use only - do not remove it.
        all other paths should only contain the file name, these paths must be
        concatenated with dir_path, for example: os.path.join(dir_path, filename).
        :param dir_path: path to the submission directory (for internal use only).
        :return:
        """
        self.model = RegNet()
        # join paths
        checkpoint_path = os.path.join(dir_path, self.checkpoint)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, input_image):
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(input_image.astype('uint8'))
        # image = cv2.resize(input_image, (512, 512))
        # image = image / 255
        img = val_transforms()(img)
        # # PIL.Image为RGB格式，替换为BGR格式
        # img = img.flip(dims=[0])
        image = img.unsqueeze(0)
        image = image.to(self.device, torch.float)
        return image
    
    def predict(self, input_image, patient_info_dict):
        """
        perform the prediction given an image and the metadata.
        input_image is a ndarray read using cv2.imread(path_to_image, 1).
        note that the order of the three channels of the input_image read by cv2 is BGR.
        :param input_image: the input image to the model.
        :param patient_info_dict: a dictionary with the metadata for the given image,
        such as {'age': 52.0, 'sex': 'male', 'height': nan, 'weight': 71.3},
        where age, height and weight are of type float, while sex is of type str.
        :return: an int value indicating the class for the input image.
        """

        image = self.preprocess(input_image)
        imagehf = self.preprocess(input_image[::-1,:,:])
        imagevf = self.preprocess(input_image[:,::-1,:])
        imagehvf = self.preprocess(input_image[::-1,::-1,:])
        with torch.no_grad():

            reg_prediction1 = self.model(image)
            reg_prediction2 = self.model(imagehf)
            reg_prediction3 = self.model(imagevf)
            reg_prediction4 = self.model(imagehvf)

            reg_prediction = (reg_prediction1 + reg_prediction2 + reg_prediction3 + reg_prediction4) / 4.0
            # 分类
            # score = (clf_prediction.argmax(dim=-1) / 8.0) - 10.
            # score = score.detach().cpu()
            # 回归
            score = approximate_value(reg_prediction.detach().cpu())

        return float(score)

if __name__ == '__main__':
    ### 保证预处理和训练时的一致
    import sys 
    sys.path.append('..')
    from utils.dataset import CustomDataset
    dataset = CustomDataset('/data/rawdata/1.Images/1.TrainingSet', '/data/mmac3/data/5-fold-label/train_val2.json',   
                                  transform=val_transforms())
    # print(dataset[0][0][:,128,128])
    # print((model().preprocess(cv2.imread('/data/cjz/mmac1/raw_data/1.Images/1.TrainingSet/mmac_task_1_train_0001.png')))[0,:,128,128])
    assert (dataset[0][0] != model().preprocess(cv2.imread('/data/rawdata/1.Images/1.TrainingSet/mmac_task_3_train_0001.png'))).sum().item()==0

    # 测试两张图 
    img = cv2.imread('/data/rawdata/1.Images/1.TrainingSet/mmac_task_3_train_0001.png') # 0.125
    patient_info_dict = {}
    model_ = model()
    model_.load('./')
    print(model_.predict(img, patient_info_dict))
    img = cv2.imread('/data/rawdata/1.Images/1.TrainingSet/mmac_task_3_train_0002.png') # 0
    patient_info_dict = {}
    model_ = model()
    model_.load('./')
    print(model_.predict(img, patient_info_dict))

