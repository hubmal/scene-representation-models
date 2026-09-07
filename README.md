# Scene representation models from scratch

Implementations of NeRF and Gaussian Splatting algoritms from scratch using PyTorch.

## NeRF

![NeRF](assets/nerf_result.gif)

## 3DGS

TODO

## Results

Due to limited computational resources, training and testing were performed on downsampled images. Specifically, for the LEGO dataset, the resolution used was 200x200.

| Method | Source      | Resolution |  PSNR  |  SSIM  | LPIPS |
|--------|-------------|:----------:|:------:|:------:|:-----:|
| NeRF   | Paper       | 800x800    | 32.54  | 0.961  | 0.050 |
| NeRF   | My results  | 800x800    | 32.08  | 0.962  | 0.022 |

## Implemented

NERF:
- [x] Core pipeline implemented and working
- [x] Positional encoding
- [x] View dependence
- [x] Hierarchical sampling
- [x] Reproduce results on the LEGO scenes
TODO:
- [ ] Support for non-synthetic scenes

Gaussian Splatting:
TODO:
- [ ] Core pipeline implemented and working
- [ ] Initialization from SfM
- [ ] Densification methods
- [ ] Custom Triton renderer
- [ ] Spherical harmonics
- [ ] Support for real-world scenes

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- [ClearML](https://clear.ml/) for experiment tracking

## How to run

Download LEGO dataset

```bash
./scripts/download_dataset.sh
```

Train NeRF model
```bash
./scripts/train_nerf.py
```

Train 3DGS model
```bash
TODO
```

## References and useful materials

- [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934) — Mildenhall et al., 2020, the original NeRF paper
- [A Beginner’s 12-Step Visual Guide to Understanding NeRF: Neural Radiance Fields for Scene Representation and View Synthesis](https://medium.com/data-science/a-12-step-visual-guide-to-understanding-nerf-representing-scenes-as-neural-radiance-fields-24a36aef909a) - Aqeel Anwar, 2025, great resource for getting started with NeRF
- [What are Intrinsic and Extrinsic Camera Parameters in Computer Vision?](https://towardsdatascience.com/what-are-intrinsic-and-extrinsic-camera-parameters-in-computer-vision-7071b72fb8ec/?gi=009c00cee7a7) - Aqeel Anwar, 2022, a solid introduction to coordinate systems in CV
- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) - Bernhard Kerbl et al., 2023, the original 3DGS paper
- [A Comprehensive Overview of Gaussian Splatting](https://medium.com/data-science/a-comprehensive-overview-of-gaussian-splatting-e7d570081362) - Kate Feingold, 2023, good starting point for understanging 3DGS