# Download and extract required models

MODEL_DIR=../pre_trained

if  [ ! -d $MODEL_DIR ] 
then
    mkdir $MODEL_DIR
fi
cd $MODEL_DIR

# Download CLIP ViT-B/16
wget https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
mv ViT-B-16.pt clip_vitbase16_pretrain.pt

# Download head and fine-tuned model
gdown https://drive.google.com/uc?id=1zv8IwSJb9X9Pxz5kvQLRMREtUvnfBtzc
unzip -j vit.zip
rm vit.zip

