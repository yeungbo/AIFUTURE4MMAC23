""" 
Copyright 2023 AIFUTURE LLC.
Code for MMAC23 challenge from AIFUTURE Lab.
"""
import os
import cv2
import torch
from torch import nn
import torchvision.models as models
import torch
import torch.nn as nn
from PIL import Image
import json
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm import tqdm
import pdb
from sklearn.metrics import r2_score, mean_absolute_error
from torchvision.models import resnet50 as net 


weights = 'DEFAULT'

norm_mean= (0.485, 0.456, 0.406) # R,G,B
norm_std= (0.229, 0.224, 0.225)
size = (720, 720) # mae_vit规定输入, dino_vit没有规定


def train_transforms():
    return transforms.Compose([    
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std) 
        ])

def val_transforms(size = (720, 720)):
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
                self.net = net(pretrained=pretrain)
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

    def preprocess(self, input_image,size=(720,720)):
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB )
        img = Image.fromarray(input_image.astype('uint8'))
        # image = cv2.resize(input_image, (512, 512))
        # image = image / 255
        img = val_transforms(size)(img)
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
        image_hf = self.preprocess(input_image[:,::-1,:])
        image_vf = self.preprocess(input_image[::-1,:,:])
        image_hvf = self.preprocess(input_image[::-1,::-1,:])

        image_640 = self.preprocess(input_image, size=(640,640))
        image_800 = self.preprocess(input_image, size=(800,800))

        with torch.no_grad():
            reg_prediction1 = self.model(image)
            reg_prediction2 = self.model(image_hf)
            reg_prediction3 = self.model(image_vf)


            # reg_prediction4 = self.model(image_640)
            # reg_prediction5 = self.model(image_800)

            reg_prediction6 = self.model(image_hvf)

            reg_prediction = (reg_prediction1 + reg_prediction2 )/2.0
            # reg_prediction = reg_prediction1
            score = approximate_value(reg_prediction.detach().cpu())

        return float(score)


def regression_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return dict(r2=r2, mae=mae)

if __name__ == '__main__':
    json_file = f'/data/mmac3/data/5-fold-label/train_val2.json'
    with open(json_file, 'r') as f:
        data = json.load(f)

    model_ = model()
    model_.load(dir_path='./')

    label_list = []
    predict_list = []
    for instance in tqdm(data):    
        path = instance['image']
        true_label = instance['spherical_equivalent']

        img = cv2.imread(f'/data/rawdata/1.Images/1.TrainingSet/{path}') 

        predict = model_.predict(img,None)
        

        label_list.append(true_label)
        predict_list.append(predict)

metric = regression_metrics(label_list, predict_list)

print('r2:', metric['r2'])
print('mae:', metric['mae'])
print("score", (metric['r2'] - metric['mae'])/2.0)


print("end")



