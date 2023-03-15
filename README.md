# Trainable Projected Gradient Method (TPGM)

 This repo implements the experiments in the paper ``Trainable Projected Gradient Method for Robust Fine-Tuning``. Specifically, we divde experiments into two categories: DomainNet experiments using ResNet and ImageNet experiments using ViT. The main difference is in how TPGM is applied.

## Create conda environment
- The environment uses Pytorch 1.7 supported on CUDA 11.x and python 3.8. 
```
cd TPGM
conda env create -f environment.yml
conda activate py38
```

## DomainNet using experiments using ResNet

- The code resides in the folder `DomainNet_ResNet_Exp`.
- Following the paper, TPGM is used at every iteration of fine-tuning.

## ImageNet experiments using ViT

- The code resides in the folder `ImageNet_ViT_Exp`.
- Following the paper, TPGM is only used at the end of fine-tuning.


