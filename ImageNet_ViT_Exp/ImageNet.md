# Trainable Projected Gradient Method (TPGM)

## Logistics
 In this section, we provide code for the ImageNet experiments in the paper. Specifically, we use CLIP pre-trained ViT-B. If you just want to look at the implementation of TPGM, please see `utils/tpgm.py`.

### A Note on TPGM for CLIP ViTs
CLIP pre-trained ViTs have shown to have good linear connectivity, i.e., a fine-tuned model can be linearly interpolated with the pre-trained model without lossing performance. Practically, we first fine-tune a CLIP pre-trained ViT without TPGM. After fine-tuning, we apply TPGM to interpolate between the fine-tuned model and the pre-trained CLIP model. The same applies to the best competing method [Wise-FT](https://github.com/mlfoundations/wise-ft).

### A Note on Computation
- The ViT-B/16 models were fine-tuned on 16 V100 GPUs (16G VRAM each) with a batch size of 32 * 16;
- The training code largely follows the code base of [Masked Autoencoders](https://github.com/facebookresearch/mae) and is done using Distributed Data-Parallel Training (DDP). Due to logistic reasons, we do not provide the training code here. 

## Prepare Dataset
### Image Data
- The code uses [ImangeNet2012](https://www.image-net.org/challenges/LSVRC/2012/),  [ImageNet-v2](https://imagenetv2.org/), [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch), [Imagenet-Adversarial](https://github.com/hendrycks/natural-adv-examples), and [ImageNet-Rendition](https://github.com/hendrycks/imagenet-r).

- We provide a script to download and extract ImageNet-v2, ImageNet-Sketch, Imagenet-Adversarial and ImageNet-Rendition. Please go to the script to change the data directory accordingly (default: `/datasets/ImageNet`). 
    ```
    . datasets/download.sh
    ```
- ImangeNet2012 requires manual download. Please make sure that the dataset is also under the same folder.
- Folder Structure
    ###
        .
        ├── imagnet-2
        ├── imagnet-a
        ├── imagnet-r
        ├── ilsvrc                 # ImangeNet2012
        │   ├── train              
        │   └── val                
        └── imagenet-s
        
### Dataset Meta Data
- We provide a script to download the labels for all datasets (train/val/test) splits. 
```
. datasets/download_meta.sh
```
    
## Download Models
- TPGM and Wise-FT require three components to run: a [CLIP pre-trained backbone](https://github.com/openai/CLIP/blob/main/clip/clip.py), a CLIP zeroshot classifier head and a fine-tuned checkpoint. We provide a script to down all three. Note that the zeroshot classifier head is extracted using the [Wise-FT repo]((https://github.com/mlfoundations/wise-ft)). The models will be placed under `../pre-trained`
    ```
    . datasets/download_models.sh
    ```


## Applying TPGM and Wise-FT 

- Applying TPGM using CLIP pre-trained ViT-B. Use `--proj_freq 1` to use TPGM. Remember to update `--data_path` to reflect changes to the location of image data.
```
python main_finetune.py --output_dir ./log/TPGM --load_pretrained ../pre_trained/ViT-B-16.pt --load_ft ../pre_trained/vit_b_checkpoint-best.pth --load_head ../pre_trained/clip_vitbase16_pretrain_head.pt --proj_freq 1 --data_path /datasets/ImageNet/
```
- Applying WiSE-FT. `--mu` sets the interpolation weight for Wise-FT.
```
python main_finetune.py --output_dir ./log/wise --load_pretrained ../pre_trained/ViT-B-16.pt --load_ft ../pre_trained/vit_b_checkpoint-best.pth --load_head ../pre_trained/clip_vitbase16_pretrain_head.pt --mu 0.5 --data_path /datasets/ImageNet/
```

## Expected Results
- We compare TPGM to Wise-FT with different interpolation ratio (denoted in the parentheses). The pre-trained model is a CLIP ViT-B/16 and its corresponding zeroshot 1000-way classifier. We also show vanilla fine-tuning (FT) and the zeroshot performance of CLIP. 

|               | ImageNet (val) | ImageNet-V2 | ImageNet-A | ImageNet-R | ImageNet-S | Average |
|---------------|----------|-------------|------------|------------|------------|---------|
| Vanilla FT    | 84.27    | 74.78       | 45.54      | 65.05      | 48.28      | 63.58   |
| Zeroshot      | 67.82    | 61.19       | 50.97      | 75.73      | 45.57      | 60.26   |
| Wise-FT (0.4) | 80.58    | 72.24       | 56.94      | 79.22      | 53.78      | 68.55   |
| Wise-FT (0.5) | 82.16    | 73.66       | 56.63      | 78.25      | 54.16      | 68.97   |
| Wise-FT (0.6) | 83.24    | 74.61       | 55.21      | 76.62      | 53.99      | 68.73   |
| Wise-FT (0.7) | 84.01    | 75.39       | 53.58      | 74.56      | 53.24      | 68.16   |
| TPGM          | 83.80    | 75.27       | 55.91      | 76.87      | 54.50      | **69.27**   |

- Note that ImageNet-A and ImageNet-R only have 200 classes. Here, we sub-sample the corresponding 200 classes during evaluation. The original paper does not sub-sample, so the results are higher here. 