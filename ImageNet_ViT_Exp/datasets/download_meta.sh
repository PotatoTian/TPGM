# Download and extract meta data

META_DIR=./datasets/imagenet 

if  [ ! -d $META_DIR ] 
then
    mkdir $META_DIR
fi
cd $META_DIR

# Download imagenet meta data
gdown https://drive.google.com/uc?id=1ZiWla--2sELO1gJaXSc6Z8PjfGerjhAi
unzip -q -j imagenet.zip
rm imagenet.zip