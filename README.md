# LucidFusion: Generating 3D Gaussians with Arbitrary Unposed Images

[Hao He](https://heye0507.github.io/)$^{\color{red}{\*}}$ [Yixun Liang](https://yixunliang.github.io/)$^{\color{red}{\*}}$, [Luozhou Wang](https://wileewang.github.io/), [Yuanhao Cai](https://github.com/caiyuanhao1998), [Xinli Xu](https://scholar.google.com/citations?user=lrgPuBUAAAAJ&hl=en&inst=1381320739207392350), [Hao-Xiang Guo](), [Xiang Wen](), [Yingcong Chen](https://www.yingcong.me)$^{\**}$

$\color{red}{\*}$: Equal contribution.
\**: Corresponding author.

[Paper PDF (Arxiv)](Coming Soon) | [Project Page (Coming Soon)]() | [Gradio Demo](Coming Soon)

---

<div align="center">
    <img src="resources/output_16.gif" width="95%"/>  
    <br>
    <p><i>Note: we compress these motion pictures for faster previewing.</i></p>
</div>

<div align=center>
<img src="resources/ours_qualitative.jpeg" width="95%"/>  
  
Examples of cross-dataset content creations with our framework, the *LucidFusion*, around **~13FPS** on A800.

</div>

## 🎏 Abstract
We present a flexible end-to-end feed-forward framework, named the *LucidDreamer*, to generate high-resolution 3D Gaussians from unposed, sparse, and arbitrary numbers of multiview images.

<details><summary>CLICK for the full abstract</summary>

> Recent large reconstruction models have made notable progress in generating high-quality 3D objects from single images. However, these methods often struggle with controllability, as they lack information from multiple views, leading to incomplete or inconsistent 3D reconstructions. To address this limitation, we introduce LucidFusion, a flexible end-to-end feed-forward framework that leverages the Relative Coordinate Map (RCM).  Unlike traditional methods linking images to 3D world thorough pose, LucidFusion utilizes RCM to align geometric features coherently across different views, making it highly adaptable for 3D generation from arbitrary, unposed images. Furthermore, LucidFusion seamlessly integrates with the original single-image-to-3D pipeline, producing detailed 3D Gaussians at a resolution of $512 \times 512$, making it well-suited for a wide range of applications.

</details>

## 🔧 Training Instructions

Our code is now released! 

### Install
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

### Pretrained Weights

Our pre-trained weights will be released soon, please check back!
Our current model loads pre-trained diffusion model weights ([Stable Diffusion 2.1] (https://huggingface.co/stabilityai/stable-diffusion-2-1/tree/main) in default) for training purposes.

## 🔥 Inference
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

## 🤗 Gradio Demo

We are currently building an online demo of LucidFusion with Gradio. It is still under development, and will coming out soon!

## 🚧 Todo

- [x] Release the Stage 2 inference codes
- [ ] Release our weights
- [ ] Release the Gardio Demo
- [ ] Release the Stage 1 and 2 training codes

## 📍 Citation 
If you find our work useful, please consider citing our paper.
```
TODO
```

## Acknowledgement
This work is built on many amazing research works and open-source projects:
- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [ZeroShape](https://github.com/zxhuang1698/ZeroShape)
- [LGM](https://github.com/3DTopia/LGM)

Thanks for their excellent work and great contribution to 3D generation area.
