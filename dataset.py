""" 
Copyright 2023 AIFUTURE LLC.
Code for MMAC23 challenge from AIFUTURE Lab.
"""
import os
import refile
from sytool import load_json_items
import json
from PIL import Image
import torch.utils.data as data
from torchvision.transforms import transforms
import numpy as np
import torch 
import pdb
import cv2

class CustomDataset(data.Dataset):
    def __init__(self, root, json_file, transform=None):
        """
        :param root: 图像所在的文件夹路径
        :param json_file: 图像标注信息的json文件路径
        :param transform: 数据预处理的方法
        """
        self.root = root
        self.transform = transform

        # 读取json文件
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        # # 获取所有图像的文件名，并根据文件名进行排序
        # self.img_names = list(self.data.keys())
        # self.img_names.sort()

    def __getitem__(self, index):

        # 获取图像文件名和回归值
        img_name = self.data[index]['image']
        target = self.data[index]['spherical_equivalent']

        patient_info_dict = {}
        for name in ['age', 'sex', 'height', 'weight']:
            if name in self.data[index].keys():
                patient_info_dict[name] = self.data[index][name]
        # 读取图像
        img_path = os.path.join(self.root, img_name)
        img = Image.open(img_path)
        # 数据预处理
        if self.transform is not None:
            img = self.transform(img)
        # PIL.Image为RGB格式，替换为BGR格式
        # img = img.flip(dims=[0])
        return img, target

    def __len__(self):
        return len(self.data)




class DcDataset(data.Dataset):
    def __init__(self, transform=None, type='train'):
        """
        :param root: 图像所在的文件夹路径
        :param json_file: 图像标注信息的json文件路径
        :param transform: 数据预处理的方法
        """

        self.transform = transform

        self.sds  = 's3://amp-data/mingtong/dataset/xxx???.sds'
        self.label_datas = load_json_items(self.sds)
        
        key = 'dian_nao_yan_guang_deng_xiao_qiu_jing'

        self.label_list=[]
        self.img_path_list = []
        for data in self.label_datas:
            if key in data['extra'].keys() and (data['extra'][key]>-20 and data['extra'][key]<10) :
                self.label_list.append(data['extra'][key])
                self.img_path_list.append(data['url'])

        length = len(self.label_list)
        if type == 'train':
            self.label_list = self.label_list[:int(0.8*length)]
            self.img_path_list = self.img_path_list[:int(0.8*length)]
        elif type == 'val':
            self.label_list = self.label_list[int(0.8*length)+1:]
            self.img_path_list = self.img_path_list[int(0.8*length)+1:]
        else:
            raise NotImplementedError


    def __getitem__(self, index):

        # 获取图像文件名和回归值
        img_path = self.img_path_list[index]
        img = refile.smart_load_image(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img.astype('uint8'))

        target = self.label_list[index]
        # 数据预处理
        if self.transform is not None:
            img = self.transform(img)
        # PIL.Image为RGB格式，替换为BGR格式
        # img = img.flip(dims=[0])
        
        return img, target

    def __len__(self):
        return len(self.img_path_list)
    

if __name__ == '__main__':

    dataset = DcDataset(transform=transforms.Compose([transforms.ToTensor()]))
    print(dataset[0])
