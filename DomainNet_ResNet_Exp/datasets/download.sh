# Download and extract all datasets under the same folder. 
# Feel free to change the data directory. Please specify the new location using `--data_path`. 

DATA_DIR=/datasets/domainnet/ # Change the data directory 
if  [ ! -d $DATA_DIR ] 
then
    mkdir -p $DATA_DIR
fi
cd $DATA_DIR

# Download clipart
echo "Downloadinng clipart"
gdown http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip
echo "Extracting clipart"
unzip -q clipart.zip
rm clipart.zip

# Download infograph
echo "Downloadinng infograph"
gdown http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip
echo "Extracting infograph"
unzip -q infograph.zip
rm infograph.zip

# Download painting
echo "Downloadinng painting"
gdown http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip
echo "Extracting painting"
unzip -q painting.zip
rm painting.zip

# Download quickdraw
echo "Downloadinng quickdraw"
gdown http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip
echo "Extracting quickdraw"
unzip -q quickdraw.zip
rm quickdraw.zip

# Download real
echo "Downloadinng real"
gdown http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip 
echo "Extracting real"
unzip -q real.zip
rm real.zip

# Download sketch
echo "Downloadinng sketch"
gdown http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip
echo "Extracting sketch"
unzip -q sketch.zip
rm sketch.zip
