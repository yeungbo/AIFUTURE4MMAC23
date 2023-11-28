""" 
Copyright 2023 AIFUTURE LLC.
Code for MMAC23 challenge from AIFUTURE Lab.
"""
import torch

# 计算ranking loss
def batch_ranking_loss(pred, label, strict=False):
    n = pred.size(0)  # 批处理大小
    pair_num = n*(n-1)/2 # 配对数
    mask = torch.triu(torch.ones(n, n))  # 创建上三角全1矩阵
    mask = mask - torch.diag(torch.diag(mask))  # 对角线元素置0
    mask = mask.bool().to(pred.device)
    pairwise_diff_label = label.unsqueeze(1) - label.unsqueeze(0)  # 计算标签之间的差值
    pairwise_diff_pred = pred.unsqueeze(1) - pred.unsqueeze(0) # 计算预测值之间的差值
    
    # 情况1 如果标签值的关系为<
    pairwise_loss_1_mask = (pairwise_diff_label < 0) & mask 
    if strict == False:
        # 如果标签值的关系为<, 那么当预测值的关系为>时,则loss不为0
        pairwise_loss_1 = torch.clamp(pairwise_diff_pred, min=0) 
    else:
        # 如果标签值的关系为<,只要预测值小的程度一样或小得更多, loss就为0
        # pairwise_loss_1 = torch.clamp(pairwise_diff_pred - pairwise_diff_label, min=0) 
        # 如果标签值的关系为<,要预测值小的程度一模一样 loss才为0
        pairwise_loss_1 = torch.abs(pairwise_diff_pred - pairwise_diff_label) 
    
    # 情况2 如果标签值的关系为>
    pairwise_loss_2_mask = (pairwise_diff_label > 0) & mask
    if strict == False:
        pairwise_loss_2 = torch.clamp(-(pairwise_diff_pred), min=0) 
    else:
        # pairwise_loss_2 = torch.clamp(-(pairwise_diff_pred - pairwise_diff_label), min=0) 
        pairwise_loss_2 = torch.abs(pairwise_diff_pred - pairwise_diff_label) 
    
    # 情况3 如果标签值的关系为=, 那么当预测值的关系为不等于时,则loss不为0
    pairwise_loss_3_mask = (pairwise_diff_label == 0) & mask
    pairwise_loss_3 = torch.abs(pairwise_diff_pred)

    ranking_loss = (pairwise_loss_1_mask * pairwise_loss_1).sum() +\
          (pairwise_loss_2_mask * pairwise_loss_2).sum() +\
          (pairwise_loss_3_mask * pairwise_loss_3).sum()
    ranking_loss = ranking_loss/pair_num
    return ranking_loss

# 示例数据和标签

if __name__ == '__main__':
    label = torch.tensor([1.0, 2.0, 3.0, 4.0])  
    pred = torch.tensor([4, 4, 4, 4])  
    # loss 为 0
    print("Ranking Loss:", batch_ranking_loss(pred, label))


    label = torch.tensor([1.0, 2.0, 3.0, 4.0])  
    pred = torch.tensor([4, 3, 2, 1])  
    # loss为10/6
    print("Ranking Loss:", batch_ranking_loss(pred, label))

    label = torch.tensor([1.0, 2.0, 3.0])  
    pred = torch.tensor([1, 3, 10])  
    # loss为0
    print("Ranking Loss:", batch_ranking_loss(pred, label))

    label = torch.tensor([1.0, 2.0, 3.0])  
    pred = torch.tensor([1, 3, -1])  
    # loss为6/3
    print("Ranking Loss:", batch_ranking_loss(pred, label))

    ############# strict 为 True ##########################
    label = torch.tensor([1.0, 2.0, 3.0, 4.0])  
    pred = torch.tensor([4, 4, 4, 4])  
    # loss 为 10/6
    print("Ranking Loss:", batch_ranking_loss(pred, label, strict=True))


    label = torch.tensor([1.0, 2.0, 3.0, 4.0])  
    pred = torch.tensor([4, 3, 2, 1])  
    # loss为 20/6
    print("Ranking Loss:", batch_ranking_loss(pred, label, strict=True))

    label = torch.tensor([1.0, 2.0, 3.0])  
    pred = torch.tensor([1, 3, 10])  
    # loss为 14/3
    print("Ranking Loss:", batch_ranking_loss(pred, label, strict=True))

    label = torch.tensor([1.0, 2.0, 3.0])  
    pred = torch.tensor([1, 3, -1])  
    # loss为 10/3
    print("Ranking Loss:", batch_ranking_loss(pred, label, strict=True))