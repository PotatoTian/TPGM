# Trainable Projected Gradient Method (TPGM)
## Logistics
 In this section, we provide code for the DomainNet experiments in the paper. Specifically, we use CLIP pre-trained ResNet50 and MOCOv3 pre-trained ResNet50. If you just want to look at the implementation of TPGM, please see `utils/tpgm.py`.


### A Note on Computation
- The following experiments were conducted on 4 RTX2080 Ti (11G VRAM) GPUs with a batch size of 64*4. 

## Prepare DomainNet
### Image Data
- We provide a script to download and extract [Clipart](http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip), [Infograph](http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip), [Painting](http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip), [Quickdraw](http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip), [Real](http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip), [Sketch](http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip) from DomainNet. Please go to the script to change the data directory accordingly (default: `/datasets/domainnet`). 
    ```
    . datasets/download.sh
    ```
    
### Dataset Meta Data
- We provide a script to download the labels for all required datasets splits. 
```
. datasets/download_meta.sh
```
    
## Download models
- We use [CLIP ResNet50](https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt) and [MocoV3 ResNet50](https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar) for the experiments. 
- The following script downloads the required pre-trained models and places them under `../pre_trained`.
    ```
    . datasets/download_models.sh
    ```
## Fine-tuning examples
- Fine-tuning CLIP ResNet50 w/o TPGM on DomainNet Real (100%). We also supprot fine-tuning on [1%,5%,10%,20%,50%] data. Please update `--percent 10` to reflect the change.  Remember to also update `--data_dir` to reflect changes to the location of image data.

    ```
    python main_finetune.py --arch clip_resnet50 --id clip_vanilla --percent 100 --lr 1e-3 --epoch 50 --gpu_per_node 4  --batch_size 64 --data_dir /datasets/domainnet --load_pretrained ../pre_trained/clip_resnet50_pretrain.pt --data_dir /datasets/domainnet
    ```

- Fine-tuning CLIP ResNet50 with TPGM on DomainNet Real (100%). 
    ```
    python main_finetune.py --arch clip_resnet50 --id clip_TPGM --percent 100 --lr 1e-2 --epoch 50 --gpu_per_node 4  --batch_size 64 --data_dir /datasets/domainnet  --load_pretrained ../pre_trained/clip_resnet50_pretrain.pt --proj_freq 1
    ```

- Fine-tuning MocoV3 ResNet50 w/o TPGM on DomainNet Real (100%). 
    ```
    python main_finetune.py --arch resnet50 --id moco_vanilla --percent 100 --lr 5e-2 --epoch 50 --gpu_per_node 4  --batch_size 64 --data_dir /datasets/domainnet  --load_pretrained ../pre_trained/mocov3_resnet50_pretrain.tar 
    ```

- Fine-tuning MocoV3 ResNet50 with TPGM on DomainNet Real (100%). 
    ```
    python main_finetune.py --arch resnet50 --id moco_TPGM --percent 100 --lr 1e-2 --epoch 50 --gpu_per_node 4  --batch_size 64 --data_dir /datasets/domainnet --load_pretrained ../pre_trained/mocov3_resnet50_pretrain.tar --proj_freq 1
    ```

## Evaluation examples and expected results
- We provide some fine-tuned models here for quick evaluation. 
- Example evalution command
    ```
    python main_eval.py --arch clip_resnet50 --gpu_per_node 4 --batch_size 128 --load_pretrained ../pre_trained/clip_resnet50_pretrain.pt --resume ./log/clip_real_100_iter_1_freq_2_25_01_2023_16_42_20/ckpt.best.pth.tar
    ```