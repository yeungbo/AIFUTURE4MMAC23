#!/usr/bin/env bash
python -m  torch.distributed.launch --nproc_per_node 8 train_ddp.py \
--epochs 100 --step_size 50 --lr 1e-4 --bs 64 \
--load_ssl_pretrain_from /data/dino/output/resnet-50-200e-v2/checkpoint_student.pt \
--freeze_stage_num 2 \
--mse_loss \
--smoothl1_loss \
--mse_loss_scale 1.0 \
--smoothl1_scale 0.0 \
--affix 'r50_v2_f2_flip'\

# python -m  torch.distributed.launch --nproc_per_node 8 train_ddp.py \
# --epochs 100 --step_size 50 --lr 1e-4 --bs 32 \
# --freeze_stage_num 0 \
# --mse_loss \
# --smoothl1_loss \
# --mse_loss_scale 1.0 \
# --smoothl1_scale 0.0 \
# --affix 'effb0_1IN1k_lr1e-4_f0'\