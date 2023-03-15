# Download and extract all datasets under the same folder. 
# Feel free to change the data directory. Please specify the new location using `--data_path`. 

DATA_DIR=/datasets/ImageNet2 # Change the data directory 
if  [ ! -d $DATA_DIR ] 
then
    mkdir $DATA_DIR
fi
cd $DATA_DIR

# Download ImageNet-V2
echo "Downloadinng ImagnetNet-V2"
wget https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz
echo "Extracting ImagnetNet-V2"
tar -xf imagenetv2-matched-frequency.tar.gz
rm imagenetv2-matched-frequency.tar.gz
mv ./imagenetv2-matched-frequency-format-val/ ./imagenet-2

# Download ImageNet-S
echo "Downloadinng ImagnetNet-S"
wget https://huggingface.co/datasets/imagenet_sketch/resolve/main/data/ImageNet-Sketch.zip
echo "Extracting ImagnetNet-S"
unzip -q ImageNet-Sketch.zip
rm ImageNet-Sketch.zip
mv ./sketch ./imagenet-s

# Download ImageNet-A
echo "Downloadinng ImagnetNet-A"
wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar
echo "Extracting ImagnetNet-A"
tar -xf imagenet-a.tar
rm imagenet-a.tar

# Download ImageNet-R
echo "Downloadinng ImagnetNet-R"
wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
echo "Extracting ImagnetNet-R"
tar -xf imagenet-r.tar
rm imagenet-r.tar