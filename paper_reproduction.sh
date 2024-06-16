#!/bin/bash
export WANDB__SERVICE_WAIT=300

# COCOStuff Vit-S
python3 src/train_segmentation.py data_dir=/mnt/hdd/leon/datasets/ decay_every_steps=250 depth_feat_correlation_loss=True depth_feat_shift=0.03 depth_feat_weight=0.19 depth_loss_decay=True depth_loss_decay_factor=0.6 depth_sampling=fps fps_sample_decay=True fps_sample_decay_every_steps=1000 fps_sample_decay_factor=0.9 neg_inter_shift=0.761 neg_inter_weight=0.7000000000000001 output_root=/mnt/hdd/leon/outputs/ pos_inter_shift=0.025 pos_inter_weight=0.36 pos_intra_shift=0.07 pos_intra_weight=0.58

# COCOStuff ViT-B
python3 src/train_segmentation.py batch_size=32 data_dir=/mnt/hdd/leon/datasets/ decay_every_steps=300 depth_feat_correlation_loss=True depth_feat_shift=0.035909146298813595 depth_feat_weight=0.16026274975444096 depth_loss_decay=True depth_loss_decay_factor=0.64 depth_sampling=fps dim=90 feature_samples=12 fps_sample_decay=True fps_sample_decay_every_steps=1000 fps_sample_decay_factor=1 model_type=vit_base neg_inter_shift=0.9748103425096648 neg_inter_weight=0.2485038032028848 output_root=/mnt/hdd/leon/outputs/ pos_inter_shift=0.21028290947990444 pos_inter_weight=1.0500945312858674 pos_intra_shift=0.12326312284078644 pos_intra_weight=0.23052367315917113 val_freq=50

# Cityscapes ViT-B
python3 src/train_segmentation.py batch_size=32 data_dir=/mnt/hdd/leon/datasets/ dataset_name=cityscapes decay_every_steps=400 depth_feat_correlation_loss=True depth_feat_shift=0.03 depth_feat_weight=0.09 depth_loss_decay=True depth_loss_decay_factor=0.8 depth_sampling=none dim=100 log_dir=cityscapes model_type=vit_base neg_inter_shift=0.26 neg_inter_weight=0.5700000000000001 output_root=/mnt/hdd/leon/outputs/ pointwise=False pos_inter_shift=0.25 pos_inter_weight=1.02 pos_intra_shift=0.39 pos_intra_weight=0.95

# Potsdam ViT-S
python3 src/train_segmentation.py batch_size=16 data_dir=/mnt/hdd/leon/datasets/ dataset_name=potsdam decay_every_steps=200 depth_feat_correlation_loss=True depth_feat_shift=0.14 depth_feat_weight=0.13 depth_loss_decay=True depth_loss_decay_factor=1 depth_sampling=fps dim=90 feature_samples=11 log_dir=potsdam model_type=vit_small neg_inter_shift=0.63 neg_inter_weight=0.72 output_root=/mnt/hdd/leon/outputs/ pointwise=True pos_inter_shift=0.09 pos_inter_weight=0.34 pos_intra_shift=0.2 pos_intra_weight=0.61