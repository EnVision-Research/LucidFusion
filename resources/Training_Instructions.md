# Preparation
This is the official implementation of *LucidFusion: Generating 3D Gaussians with Arbitrary Unposed Images*.

## Install
```
# Xformers is required! please refer to https://github.com/facebookresearch/xformers for details.
# For example, we use torch 2.3.1 + cuda 11.8
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
# [linux only] cuda 11.8 version
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118

# For 3D Gaussian Splatting, we use the official installation. Please refer to https://github.com/graphdeco-inria/gaussian-splatting for details
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
pip install ./diff-gaussian-rasterization

# Other dependencies
conda env create -f environment.yml
conda activate LucidFusion
```

## Pretrained Weights

Our pre-trained weights will be released soon, please check back!
Our current model loads pre-trained diffusion model weights ([Stable Diffusion 2.1] (https://huggingface.co/stabilityai/stable-diffusion-2-1/tree/main) in default) for training purposes.

## Inference
A shell script is provided with example files.
To run, you first need to setup the pretrained weights as follows:

```
cd LucidFusion
mkdir output/demo

# Download the pretrained weights and name it as best.ckpt

# Place the pretrained weights in LucidFusion/output/demo/best.ckpt

# The result will be saved in the LucidFusion/output/demo/<obj name>/

bash scripts/demo.sh
```

More example files will be released soon

You can also try your own example! To do that:

First obtain images and place them in the examples folder
```
LucidFusion
├── examples/
|   ├── "your obj name"/
|   |   ├── "image_01.png"
|   |   ├── "image_02.png"
|   |   ├── ...
```
Then you need to run preprocess.py to extract recentered image and its mask:
```
# Run the following will create two folders (images, masks) in "your obj name" folder.
# You can check to see if the extract mask is corrected.
python preprocess.py "examples/"you obj name" --outdir examples/"your obj name"

# Modify demo.sh to DEMO='examples/"you obj name" ' then run the file
bash scripts/demo.sh
```
