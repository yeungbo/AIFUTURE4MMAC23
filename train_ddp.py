""" 
Copyright 2023 AIFUTURE LLC.
Code for MMAC23 challenge from AIFUTURE Lab.
"""
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import random
from models import RegNet, approximate_value, clf
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import datetime
import matplotlib.pyplot as plt
from utils import (
    get_trainloader,
    get_valloader,
    distributed_concat,
    batch_ranking_loss,
    batch_pearsonr_loss,
    regression_metrics,
)
import numpy as np
from loguru import logger
import warnings

warnings.filterwarnings("ignore")

###  初始化我们的模型、数据、各种配置  ####
parser = argparse.ArgumentParser()
# DDP：从外部得到local_rank参数, 不用自己给, -m torch.distributed.launch会自动给
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--bs", type=int, default=32, help="batch size")
parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
parser.add_argument("--step_size", type=int, default=50, help="step size of StepLR")
parser.add_argument("--step_gamma", type=float, default=0.1, help="gamma of StepLR")
parser.add_argument("--resume_from", type=str, default=None)
parser.add_argument("--load_ssl_pretrain_from", type=str, default=None)
parser.add_argument("--freeze_stage_num", type=int, default=2)
parser.add_argument("--mse_loss", action="store_true")
parser.add_argument("--smoothl1_loss", action="store_true")
parser.add_argument("--mse_loss_scale", type=float, default=1.0)
parser.add_argument("--smoothl1_scale", type=float, default=1.0)
parser.add_argument("--affix", type=str, default="")
parser.add_argument("--cosine_lr", action="store_true")
args = parser.parse_args()
local_rank = args.local_rank


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(66666)


# DDP：DDP backend初始化
torch.cuda.set_device(local_rank)
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12345'
dist.init_process_group(backend="nccl",rank=0,world_size=1)  # nccl是GPU设备上最快、最推荐的后端


def train(
    args,
    save_folder,
    train_label_path="data/train_train.json",
    val_label_path="data/train_val.json",
):
    # 准备数据，要在DDP初始化之后进行
    trainloader, valloader = get_trainloader(
        args.bs, train_label_path=train_label_path, type="compe"
    ), get_valloader(args.bs, val_label_path=val_label_path, type="compe")

    # 构造模型
    model = RegNet(pretrain=True, clf=clf)
    # 引入SyncBN，这句代码，会将普通BN替换成SyncBN。
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(local_rank)
    # DDP: Load模型要在构造DDP模型之前，且只需要在master上加载就行了。
    ckpt_path = args.resume_from
    if dist.get_rank() == 0 and ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path))
        print("load resume successfully")

    # 加载自监督预训练模型
    if args.load_ssl_pretrain_from is not None:
        if dist.get_rank() == 0:
            msg = model.net.load_state_dict(
                torch.load(args.load_ssl_pretrain_from), strict=False
            )
            print(msg)
    if args.freeze_stage_num > 0:
        print("start freeze")
        # resnet: 冻结前args.freeze_stage_num个stage
        for name, param in model.net.named_parameters():
            if any(
                [name.startswith(f"layer{i}") for i in range(args.freeze_stage_num + 1)]
            ):
                # print(f'freeze layer: {name}')
                param.requires_grad = False
        # dino_vit:
        for name, param in model.net.named_parameters():
            if any(
                [
                    name.startswith(f"blocks.{i}.")
                    for i in range(args.freeze_stage_num + 1)
                ]
            ):
                # print(f'freeze layer: {name}')
                param.requires_grad = False
    # DDP: 构造DDP model
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # DDP: 要在构造DDP model之后，才能用model初始化optimizer。
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    milestones = [50, 75]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=args.step_gamma
    )

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs, eta_min=0)

    # loss
    mse_func = nn.MSELoss().to(local_rank)
    smoothl1_func = nn.SmoothL1Loss(beta=1.0).to(local_rank)
    ce_func = nn.CrossEntropyLoss().to(local_rank)
    ###  网络训练  ###
    train_losses = []
    val_losses = []
    lrs = []
    metrics_list = []
    best_model_path = None
    best_r2_path = None
    best_mae_path = None
    metrics_best = -float("inf")
    metrics_best_r2 = -float("inf")
    metrics_best_mae = float("inf")
    metrics_best_dict = {}

    for epoch in range(args.epochs):
        if dist.get_rank() == 0:
            print("#" * 100)
            print(f"epoch{epoch}/{args.epochs}: ")
        now = datetime.datetime.now()
        model.train()
        # DDP：设置sampler的epoch，
        # DistributedSampler需要这个来指定shuffle方式，
        # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果。
        # ddp里面，设置了dataloader的sampler，并且通过下面这行达到shuffle效果，
        # 不能再设定dataloader的shuffle参数为True
        trainloader.sampler.set_epoch(epoch)
        train_loss_s = []
        mse_loss_s = []
        smoothl1_loss_s = []

        if dist.get_rank() == 0:  # ddp中避免出现多条进度条的tqdm用法
            pbar = tqdm(total=len(trainloader), desc="lter", position=0)
        for data, label in trainloader:
            data, label = data.to(local_rank), label.to(local_rank)
            if clf:
                reg_output, clf_output = model(data)
            else:
                reg_output = model(data)
            reg_output = reg_output.squeeze(dim=-1)

            float_label = label.to(torch.float)
            if args.mse_loss:
                mse_loss = mse_func(reg_output, float_label)
            else:
                mse_loss = 0.0
            if args.smoothl1_loss:
                smoothl1_loss = smoothl1_func(reg_output, float_label)
            else:
                smoothl1_loss = 0.0
            loss = args.mse_loss_scale * mse_loss + args.smoothl1_scale * smoothl1_loss

            optimizer.zero_grad()
            loss.backward()

            train_loss_s.append(loss * data.size(0))
            if args.mse_loss:
                mse_loss_s.append(mse_loss * data.size(0))
            if args.smoothl1_loss:
                smoothl1_loss_s.append(smoothl1_loss * data.size(0))
            if dist.get_rank() == 0:
                pbar.update(1)
            optimizer.step()

        # barrier 的功能，所有进程都运行到dist.barrier()，才开始运行下一行
        # dist.barrier()
        # 方式1：用all_reduce
        train_loss_s = torch.tensor(train_loss_s).to(local_rank)
        dist.all_reduce(train_loss_s, op=dist.ReduceOp.SUM)
        # 方式2：用all_gather
        # train_loss_s = distributed_concat(torch.tensor(train_loss_s).to(local_rank))
        mse_loss_s = distributed_concat(torch.tensor(mse_loss_s).to(local_rank))
        smoothl1_loss_s = distributed_concat(
            torch.tensor(smoothl1_loss_s).to(local_rank)
        )

        train_loss_epoch = (train_loss_s.sum() / len(trainloader.dataset)).item()
        mse_loss_epoch = (mse_loss_s.sum() / len(trainloader.dataset)).item()
        smoothl1_loss_epoch = (smoothl1_loss_s.sum() / len(trainloader.dataset)).item()

        # Adjust the learning rate
        scheduler.step()

        ###  每个epoch进行网络验证  ###
        model.eval()
        val_loss = 0.0
        predictions = []
        clf_output_concat = []
        labels = []
        with torch.no_grad():
            for data, label in valloader:
                data, label = data.to(local_rank), label.to(local_rank)
                if clf:
                    reg_output, clf_output = model(data)
                else:
                    reg_output = model(data)
                reg_output = reg_output.squeeze(dim=-1)
                predictions.append(approximate_value(reg_output))
                labels.append(label)
            # 进行gather
            predictions = distributed_concat(torch.concat(predictions, dim=0))
            labels = distributed_concat(torch.concat(labels, dim=0))

            mse_loss = 0
            float_labels = labels.to(torch.float)

            if args.mse_loss:
                mse_loss = mse_func(predictions, float_labels)
            if args.smoothl1_loss:
                smoothl1_loss = smoothl1_func(predictions, float_labels)
            val_loss = (
                args.mse_loss_scale * mse_loss + args.smoothl1_scale * smoothl1_loss
            )
            val_loss = val_loss.item()
            metrics_dict = regression_metrics(
                np.array(labels.detach().to("cpu")), predictions.detach().to("cpu")
            )
            metrics = (metrics_dict["r2"] - metrics_dict["mae"]) / 2  # r2越大越好，mae越小越好
            # 在进程里打印日志, 保存模型
            if dist.get_rank() == 0:
                print(f"完全正确：{(labels == predictions).sum()}/{labels.shape[0]}")
                print(f"train_loss={train_loss_epoch:.3f} ", end="")
                if args.mse_loss:
                    print(f"mse_loss={mse_loss_epoch:.3f} ", end="")
                if args.smoothl1_loss:
                    print(f"smothl1loss={smoothl1_loss_epoch:.3f} ", end="")
                print(f"val_loss={val_loss:.3f} ", end="")
                print(f"metrics={metrics:.3f} ", end="")
                print("metrics_best r2", metrics_dict["r2"], end=" ")
                print("metrics_best -mae", -metrics_dict["mae"], end=" ")
                print(f"lr={optimizer.param_groups[0]['lr']:.5f}")
                if metrics > metrics_best:
                    metrics_best_dict = metrics_dict
                    metrics_best = metrics
                    if best_model_path:
                        os.remove(best_model_path)
                    best_model_path = os.path.join(
                        save_folder,
                        f"best_model_epoch{epoch}_metrics_{round(metrics, 5)}.pt",
                    )
                    # 1. save模型的时候，和DP模式一样，有一个需要注意的点：
                    #    保存的是model.module而不是model。
                    #    因为model其实是DDP model，参数是被`model=DDP(model)`包起来的。
                    # 2. 只需要在进程0上保存一次就行了，避免多次保存重复的东西。
                    torch.save(model.module.state_dict(), best_model_path)

                if metrics_dict["r2"] > metrics_best_r2:
                    metrics_best_r2 = metrics_dict["r2"]
                    if best_r2_path:
                        os.remove(best_r2_path)
                    best_r2_path = os.path.join(
                        save_folder,
                        f"best_r2_model_{epoch}epoch_{round(metrics_best_r2, 5)}.pt",
                    )
                    torch.save(model.module.state_dict(), best_r2_path)
                if metrics_dict["mae"] < metrics_best_mae:
                    metrics_best_mae = metrics_dict["mae"]
                    if best_mae_path:
                        os.remove(best_mae_path)
                    best_mae_path = os.path.join(
                        save_folder,
                        f"best_mae_model_{epoch}epoch_{round(metrics_best_mae,5)}.pt",
                    )
                    torch.save(model.module.state_dict(), best_mae_path)

                if (epoch in [75, 80, 85, 90]) or (epoch + 1) == args.epochs:
                    # if (epoch + 1) % (max(args.epochs//4, 1)) == 0 or (epoch + 1) == args.epochs: # 每训1/4保存一次模型
                    model_path = os.path.join(
                        save_folder,
                        f"model_epoch{epoch}_metrics_{round(metrics, 5)}.pt",
                    )
                    torch.save(model.module.state_dict(), model_path)

                train_losses.append(train_loss_epoch)
                val_losses.append(val_loss)
                lrs.append(optimizer.param_groups[0]["lr"])
                metrics_list.append(metrics)
        time = datetime.datetime.now()
        if dist.get_rank() == 0:
            logger.info(time - now)
    # 保存可视化loss图和训练相关文件到log文件夹
    if dist.get_rank() == 0:
        # 可视化损失和学习率
        fig, ax1 = plt.subplots()
        # 绘制训练集loss和验证集loss
        ax1.plot(train_losses, label="train_loss")
        ax1.plot(val_losses, label="val_loss")
        ax1.set_xlabel("epochs")
        ax1.set_ylabel("loss")
        ax1.legend(loc="upper left")
        # 绘制指标
        ax2 = ax1.twinx()
        ax2.plot(metrics_list, label="metrics", color="red")
        ax2.set_ylabel("metrics")
        ax2.legend(loc="upper right")
        # # 设置标题
        plt.title("Training Curve")
        plt.savefig(os.path.join(save_folder, "loss_metrics.jpg"))
    return metrics_best, metrics_best_dict


# 普通训练
def normal_train(k=2):
    now = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
    save_folder = os.path.join("log", f"normal+{now}_{args.affix}")

    if dist.get_rank() == 0:
        print("+" * 50 + f"fold{k}" + "+" * 50)
    save_folder_k = os.path.join(save_folder, f"{k}")
    os.makedirs(save_folder_k, exist_ok=True)
    # 保存model、utils文件夹、train_ddp文件
    if dist.get_rank() == 0:
        if k == 1:
            os.system(f"cp train_ddp.py {save_folder_k}")
            os.system(f"cp train.sh {save_folder_k}")
            os.system(f"cp -r models {save_folder_k}")
            os.system(f"cp -r utils {save_folder_k}")
        else:
            os.system(
                f"cp {os.path.join(save_folder, f'{1}', 'train_ddp.py')} {save_folder_k}"
            )
            os.system(
                f"cp {os.path.join(save_folder, f'{1}', 'train.sh')} {save_folder_k}"
            )
            os.system(
                f"cp -r {os.path.join(save_folder, f'{1}', 'models')} {save_folder_k}"
            )
            os.system(
                f"cp -r {os.path.join(save_folder, f'{1}', 'utils')} {save_folder_k}"
            )
    metrics_best, metrics_best_dict = train(
        args,
        save_folder=save_folder_k,
        train_label_path=f"data/5-fold-label/train_train{k}.json",
        val_label_path=f"data/5-fold-label/train_val{k}.json",
    )
    if dist.get_rank() == 0:
        with open(os.path.join(save_folder, "log.log"), mode="w") as f:
            print("metrics_best r2", metrics_best_dict["r2"], "\n", file=f)
            print("metrics_best -mae", -metrics_best_dict["mae"], "\n", file=f)
            # print('val_loss_best spe mean', metrics_best_dict['spe'], '\n', file=f)
            print("metrics_best:", metrics_best, file=f)
            now = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
            print("now:", now, file=f)
        print("metrics_best r2", metrics_best_dict["r2"], "\n")
        print("metrics_best -mae", -metrics_best_dict["mae"], "\n")


# 五折训练
def five_fold_train():
    now = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
    save_folder = os.path.join("log", f"five{now}_{args.affix}")

    metrics_best_s = []
    metrics_best_r2_s = []
    metrics_best_mae_s = []
    for k in range(1, 6):
        if dist.get_rank() == 0:
            print("+" * 50 + f"fold{k}" + "+" * 50)
        save_folder_k = os.path.join(save_folder, f"{k}")
        os.makedirs(save_folder_k, exist_ok=True)
        # 保存model、utils文件夹、train_ddp文件
        if dist.get_rank() == 0:
            if k == 1:
                os.system(f"cp train_ddp.py {save_folder_k}")
                os.system(f"cp train.sh {save_folder_k}")
                os.system(f"cp -r models {save_folder_k}")
                os.system(f"cp -r utils {save_folder_k}")
            else:
                os.system(
                    f"cp {os.path.join(save_folder, f'{1}', 'train_ddp.py')} {save_folder_k}"
                )
                os.system(
                    f"cp {os.path.join(save_folder, f'{1}', 'train.sh')} {save_folder_k}"
                )
                os.system(
                    f"cp -r {os.path.join(save_folder, f'{1}', 'models')} {save_folder_k}"
                )
                os.system(
                    f"cp -r {os.path.join(save_folder, f'{1}', 'utils')} {save_folder_k}"
                )
        metrics_best, metrics_best_dict = train(
            args,
            save_folder=save_folder_k,
            train_label_path=f"data/5-fold-label/train_train{k}.json",
            val_label_path=f"data/5-fold-label/train_val{k}.json",
        )
        if dist.get_rank() == 0:
            metrics_best_s.append(metrics_best)
            metrics_best_r2_s.append(metrics_best_dict["r2"])
            metrics_best_mae_s.append(metrics_best_dict["mae"])
            # metrics_best_spe_s.append(metrics_best_dict['spe'])
            print(metrics_best_s)
    if dist.get_rank() == 0:
        with open(os.path.join(save_folder, "log.log"), mode="w") as f:
            print(
                "5 fold metrics_best r2 mean", np.mean(metrics_best_r2_s), "\n", file=f
            )
            print(
                "5 fold metrics_best -mae mean",
                -np.mean(metrics_best_mae_s),
                "\n",
                file=f,
            )
            # print('5 fold val_loss_best spe mean', np.mean(metrics_best_spe_s), '\n', file=f)
            print("5 fold metrics_best mean", np.mean(metrics_best_s), "\n", file=f)
            print("5 fold metrics_best:", metrics_best_s, file=f)
            now = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
            print("now:", now, file=f)


# 所有数据用来训练
def all_train():
    now = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
    save_folder = os.path.join("log", f"all{now}_{args.affix}")
    save_folder_all = os.path.join(save_folder, f"all")
    os.makedirs(save_folder_all, exist_ok=True)
    # 保存model、utils文件夹、train_ddp文件
    if dist.get_rank() == 0:
        os.system(f"cp train_ddp.py {save_folder_all}")
        os.system(f"cp train.sh {save_folder_all}")
        os.system(f"cp -r models {save_folder_all}")
        os.system(f"cp -r utils {save_folder_all}")
    metrics_best, metrics_best_dict = train(
        args,
        save_folder=save_folder_all,
        train_label_path=f"data/train.json",
        val_label_path=f"data/train.json",
    )
    if dist.get_rank() == 0:
        with open(os.path.join(save_folder, "log.log"), mode="w") as f:
            print("metrics_best r2", metrics_best_dict["r2"], "\n", file=f)
            print("metrics_best -mae", -metrics_best_dict["mae"], "\n", file=f)
            # print('val_loss_best spe mean', metrics_best_dict['spe'], '\n', file=f)
            print("metrics_best:", metrics_best, file=f)
            now = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
            print("now:", now, file=f)


if __name__ == "__main__":
    normal_train()

