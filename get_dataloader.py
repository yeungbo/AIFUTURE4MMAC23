""" 
Copyright 2023 AIFUTURE LLC.
Code for MMAC23 challenge from AIFUTURE Lab.
"""
import torchvision.transforms as transforms
from .dataset import CustomDataset, DcDataset
from torch.utils.data.distributed import DistributedSampler
import torch
from torch.utils.data import DataLoader
from .ddp import SequentialDistributedSampler
from models.model import val_transforms, train_transforms

def get_dataloader(dataset, batch_size, num_workers, ddp:bool, shuffle:bool, pin_memory:bool):
    # num_gpu = torch.cuda.device_count() # 可见gpu数，可能会大于目前ddp所用的gpu数
    if ddp:
        world_size = torch.distributed.get_world_size()
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size//world_size, 
                                num_workers=num_workers, sampler=sampler,
                                shuffle=shuffle,pin_memory=pin_memory)
    else:
        dataloader = DataLoader(dataset, batch_size,num_workers=num_workers, 
                                shuffle=shuffle, pin_memory=pin_memory)
    return dataloader

def get_trainloader(batch_size=32, num_workers=2, ddp=True, 
                    shuffle=False, pin_memory=True,
                    img_dir='/data/rawdata/1.Images/1.TrainingSet',
                    train_label_path='data/train_train.json',
                    type='compe'
                    ):
    transform = train_transforms()
    if type == 'compe':
        print("use competition data")
        train_dataset = CustomDataset(root=img_dir, json_file=train_label_path,
                                    transform=transform)
    else:
        print("use dongcheng data")
        train_dataset = DcDataset(transform=transform, type='train')
    return get_dataloader(train_dataset, batch_size=batch_size, 
                          num_workers=num_workers, ddp=ddp, shuffle=shuffle,
                          pin_memory=pin_memory)

def get_valloader(batch_size=32, num_workers=2, ddp=True,
                  img_dir='/data/rawdata/1.Images/1.TrainingSet',
                  val_label_path='data/train_val.json',
                  shuffle=False, pin_memory=True, type='compe'
                  ):
    transform = val_transforms()
    if type == 'compe':
        print("use competition data")
        val_dataset = CustomDataset(img_dir, val_label_path,   
                                    transform=transform)
    else:
        print("use dongchneg data")
        val_dataset = DcDataset(transform=transform, type='val')
    return get_dataloader(val_dataset, batch_size=batch_size, 
                          num_workers=num_workers, ddp=ddp,
                          shuffle=shuffle, pin_memory=pin_memory)


def get_valloader_sequential(batch_size=32,
                  img_dir='/data/rawdata/1.Images/1.TrainingSet',
                  val_label_path='data/train_val.json'
                  ):
    transform = val_transforms()
    val_dataset = CustomDataset(img_dir, val_label_path,   
                                  transform=transform)
    # return get_dataloader(val_dataset, batch_size=batch_size, 
    #                       num_workers=num_workers, ddp=ddp)
    test_sampler = SequentialDistributedSampler(val_dataset, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler=test_sampler)
    return testloader