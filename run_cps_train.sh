mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
apt install lsof iproute2 procps -y

source ~/miniconda3/bin/activate

git clone https://github.com/PetterPet01/TorchSemiSeg
cd TorchSemiSeg

pip install archspec
pip install charset-normalizer
conda install archspec

conda create -n semiseg python=3.6
conda activate semiseg
conda env update --file semiseg.yaml

cd ./furnace/apex
python setup.py install --cpp_ext --cuda_ext

# pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 "typing-extensions==3.10.0.2" -f https://download.pytorch.org/whl/torch_stable.html
# Step 1: Install compatible typing-extensions
pip install "typing-extensions==3.10.0.2" # Or 3.7.4.3

# Step 2: Install PyTorch, torchvision, torchaudio for PyTorch 1.7.1 and CUDA 11.0
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html


cd ../../
cd ./DATA
rm -rf city pascal_voc

gdown 'https://drive.google.com/uc?id=1S16tgrk-goEKvymIYg-sQVGWlM9ooaCW' -O city.zip
unzip city.zip
gdown 'https://drive.google.com/uc?id=1XUUKTD4dUgXAG1nKZA4X_y3NXyvfWgcC' -O pascal_voc.zip
unzip pascal_voc.zip
gdown 'https://drive.google.com/uc?id=1YcHQY906TwNVycQ7lYRDr5mwWdYKiJ8k' -O pytorch-weight.zip
unzip pytorch-weight.zip

cd ../
cd ./exp.voc/voc8.res50v3+.CPS