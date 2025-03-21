# LucidFusion: Reconstructing 3D Gaussians with Arbitrary Unposed Images

[Hao He](https://heye0507.github.io/)$^{\*}$ [Yixun Liang](https://yixunliang.github.io/)$^{\*}$, [Luozhou Wang](https://wileewang.github.io/), [Yuanhao Cai](https://github.com/caiyuanhao1998), [Xinli Xu](https://scholar.google.com/citations?user=lrgPuBUAAAAJ&hl=en&inst=1381320739207392350), [Hao-Xiang Guo](), [Xiang Wen](), [Yingcong Chen](https://www.yingcong.me)$^{\**}$

\*: Equal contribution.
\**: Corresponding author.

[Paper PDF (Arxiv)](https://arxiv.org/abs/2410.15636) | [Project Page](https://heye0507.github.io/LucidFusion_page/) | [Model Weights](https://huggingface.co/heye0507/LucidFusion) | [Gradio Demo](Coming Soon) 

---

<p align="center">
  <img src="resources/res_ironman.gif" width="24%" alt="Ironman">
  <img src="resources/res_hulk.gif" width="24%" alt="Hulk">
  <img src="resources/res_deadpool.gif" width="24%" alt="Deadpool">
  <img src="resources/res_team_america.gif" width="24%" alt="Team America">
</p>
<p align="center">
  <img src="resources/res_venom_1.gif" width="24%" alt="Venom">
  <img src="resources/res_black_widow.gif" width="24%" alt="Black Widow">
  <img src="resources/res_spiderman.gif" width="24%" alt="Spiderman">
  <img src="resources/res_superman.gif" width="24%" alt="Superman">
</p>
<p align="center">
  <img src="resources/res_minions.gif" width="24%" alt="Minions">
  <img src="resources/res_snowman.gif" width="24%" alt="Snowman">
  <img src="resources/res_d2_witch.gif" width="24%" alt="Diablo 2">
  <img src="resources/res_harry_porter.gif" width="24%" alt="Harry Porter">
</p>
<p align="center">
  <img src="resources/princess.gif" width="24%" alt="Princess">
  <img src="resources/res_arabic.gif" width="24%" alt="Arabic">
  <img src="resources/res_chief.gif" width="24%" alt="Chief">
  <img src="resources/res_knight.gif" width="24%" alt="Kinght">
</p>
<p align="center">
  <img src="resources/res_cry_witch.gif" width="24%" alt="Witch">
  <img src="resources/boy_running.gif" width="24%" alt="Boy">
  <img src="resources/girl_head_3.gif" width="24%" alt="Girl Head">
  <img src="resources/girl_head_2.gif" width="24%" alt="Girl Head 2">
</p>



<div align="center">
    <img src="resources/output_16.gif" width="95%"/>  
    <br>
    <p><i>Note: we compress these motion pictures for faster previewing.</i></p>
</div>

## 📢 News
- 2024-11-08: We added 3 new preprocessed example cases for 256x256 resolution inputs. You can now run "dr_strange", "superman", and "minions_stuart" using demo.sh <br>
- 2024-11-04: LucidFusion now supports 512x512 resolution inputs. Demo results released, and we will release the model soon!   <br>




## 🎏 Abstract
We present a flexible end-to-end feed-forward framework, named the *LucidFusion*, to reconstruct high-resolution 3D Gaussians from unposed, sparse, and arbitrary numbers of multiview images.

<details><summary>CLICK for the full abstract</summary>

> Recent large reconstruction models have made notable progress in generating high-quality 3D objects from single images. However, current reconstruction methods often rely on explicit camera pose estimation or fixed viewpoints, restricting their flexibility and practical applicability. We reformulate 3D reconstruction as image-to-image translation and introduce the Relative Coordinate Map (RCM), which aligns multiple unposed images to a “main” view without pose estimation. While RCM simplifies the process, its lack of global 3D supervision can yield noisy outputs. To address this, we propose Relative Coordinate Gaussians (RCG) as an extension to RCM, which treats each pixel’s coordinates as a Gaussian center and employs differentiable rasterization for consistent geometry and pose recovery. Our LucidFusion framework handles an arbitrary number of unposed inputs, producing robust 3D reconstructions within seconds and paving the way for more flexible, pose-free 3D pipelines.

</details>

## 🔧 Training Instructions

Our inference code is now released! We will release our training code soon!

### Install
```
conda create -n LucidFusion python=3.9.19
conda activate LucidFusion

# For example, we use torch 2.3.1 + cuda 11.8, and tested with latest torch (2.4.1) which works with the latest xformers (0.0.28).
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

# Xformers is required! please refer to https://github.com/facebookresearch/xformers for details.
# [linux only] cuda 11.8 version
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118

# For 3D Gaussian Splatting, we use LGM modified version, details please refer to https://github.com/3DTopia/LGM
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# Other dependencies
pip install -r requirements.txt
```

### Pretrained Weights
Our pre-trained weight is now released! Please check [weights](https://huggingface.co/heye0507/LucidFusion).

To download the pre-trained weights, simply run
```
python download.py
```


## 🔥 Inference
A shell script is provided with example files. Please make sure pre-trained weights is downloaded in the "pretrained" folder
```
cd LucidFusion
mkdir output/demo
```
We have also provided some preprocessed examples.

For GSO files, the example objects are "alarm", "chicken", "hat", "lunch_bag", "mario", and "shoe1".

To run GSO demo:
```
# You can adjust "DEMO" field inside the gso_demo.sh to load other examples.

bash scripts/gso_demo.sh
```

To run the images demo, masks are obtained using preprocess.py. The example objects are "nutella_new", "monkey_chair", "dog_chair".

```
bash scripts/demo.sh
```

To run the diffusion demo as a single-image-to-multi-view setup, we use the pixel diffusion trained in the CRM, as described in the paper. You can also use other multi-view diffusion models to generate multi-view outputs from a single image.

For dependencies issue, please check https://github.com/thu-ml/CRM

We also provide LGM's imagegen diffusion, simply set --crm=false in diffusion_demo.sh. You can change the --seed with different seed option.

```
bash script/diffusion_demo.sh
```


You can also try your own example! To do that:

1. Obtain images and place them in the examples folder:
```
LucidFusion
├── examples/
|   ├── "your obj name"/
|   |   ├── "image_01.png"
|   |   ├── "image_02.png"
|   |   ├── ...
```
2. Run preprocess.py to extract the recentered image and its mask:
```
# Run the following will create two folders (images, masks) in "your-obj-name" folder.
# You can check to see if the extract mask is corrected.
python preprocess.py examples/you-obj-name --outdir examples/your-obj-name
```

3. Modify demo.sh to set DEMO=“examples/your-obj-name”, then run the script:
```
bash scripts/demo.sh
```

## 🤗 Gradio Demo

For Gradio Demo test version, simply run
```
python app.py
```

Please note this demo is still under development, and check back later for the full version!

## 🚧 Todo

- [x] Release the inference codes
- [x] Release our weights
- [ ] Release our high resolution input model weights
- [ ] Release the Gardio Demo
- [ ] Release the Stage 1 and 2 training codes

## 📍 Citation 
If you find our work useful, please consider citing our paper.
```
@misc{he2024lucidfusion,
      title={LucidFusion: Generating 3D Gaussians with Arbitrary Unposed Images}, 
      author={Hao He and Yixun Liang and Luozhou Wang and Yuanhao Cai and Xinli Xu and Hao-Xiang Guo and Xiang Wen and Yingcong Chen},
      year={2024},
      eprint={2410.15636},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.15636}, 
}
```

## 💼 Acknowledgement
This work is built on many amazing research works and open-source projects:
- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [ZeroShape](https://github.com/zxhuang1698/ZeroShape)
- [LGM](https://github.com/3DTopia/LGM)

Thanks for their excellent work and great contribution to 3D generation area.
