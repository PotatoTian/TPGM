# Download and extract required models

MODEL_DIR=../pre_trained

if  [ ! -d $MODEL_DIR ] 
then
    mkdir $MODEL_DIR
fi
cd $MODEL_DIR

# Download CLIP ResNet50 
wget https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt
mv RN50.pt clip_resnet50_pretrain.pt

# Download MocoV3 ResNet50
wget https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar
mv r-50-1000ep.pth.tar mocov3_resnet50_pretrain.tar 

