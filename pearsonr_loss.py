""" 
Copyright 2023 AIFUTURE LLC.
Code for MMAC23 challenge from AIFUTURE Lab.
"""
import torch

def batch_pearsonr_loss(label, pred, epsilon=1e-8):
    # Calculate means
    mean_label = torch.mean(label)
    mean_pred = torch.mean(pred)
    # Calculate standard deviations
    std_label = torch.std(label, unbiased=False) + epsilon
    std_pred = torch.std(pred, unbiased=False) + epsilon
    # Calculate covariance
    covariance = torch.sum((label - mean_label) * (pred - mean_pred))
    # Calculate Pearson correlation coefficient
    pearson = covariance / ((std_label * std_pred)*label.shape[0])
    # Convert Pearson correlation coefficient to a loss
    loss = 1.0 - pearson
    return loss

if __name__ == "__main__":
    # 生成一些示例数据
    label = torch.tensor([1.0, 2.0, 3.0, 4.0])
    pred = torch.tensor([2.1, 4., 6., 8.])

    # 计算损失
    loss = batch_pearsonr_loss(label, pred)

    # 打印结果
    print(loss)

    from scipy.stats import pearsonr

    print(1-pearsonr(pred, label)[0])