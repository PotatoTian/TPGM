# Download and extract meta data

META_DIR=./datasets/domainnet 

if  [ ! -d $META_DIR ] 
then
    mkdir -p $META_DIR
fi
cd $META_DIR

# Download imagenet meta data
gdown https://drive.google.com/uc?id=1gmUPsiRnE78FCMkPD77elSJkoXU1vcJp
unzip -q -j domainnet.zip
rm domainnet.zip
