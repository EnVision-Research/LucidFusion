# LucidFusion: Generating 3D Gaussians with Arbitrary Unposed Images

[Hao He](https://heye0507.github.io/)$^{\color{red}{\*}}$ [Yixun Liang](https://yixunliang.github.io/)$^{\color{red}{\*}}$, [Luozhou Wang](https://wileewang.github.io/), [Yuanhao Cai](https://github.com/caiyuanhao1998), [Xinli Xu](https://scholar.google.com/citations?user=lrgPuBUAAAAJ&hl=en&inst=1381320739207392350), [Hao-Xiang Guo](), [Xiang Wen](), [Yingcong Chen](https://www.yingcong.me)$^{\**}$

$\color{red}{\*}$: Equal contribution.
\**: Corresponding author.

[Paper PDF (Arxiv)](Coming Soon) | [Project Page (Coming Soon)]() | [Gradio Demo](Coming Soon)

---

<div align=center>
TODO: GIF 

Note: we compress these motion pictures for faster previewing.
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

Our code is now released! Please refer to this [**link**](resources/Training_Instructions.md) for detailed training instructions.

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
