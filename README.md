# The official implementation of "Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation of Road Scenes"
 
![avatar](./figs/performance.png)
**Achieve state-of-the-art trade-off between accuracy and speed on cityscapes and camvid, without using inference acceleration and extra data!** 

![avatar](./figs/DDRNet_seg.png)
The overall architecture of our methods.

![avatar](./figs/DAPPM.png)
The details of "Deep Aggregation Pyramid Pooling Module (DAPPM).

# Usage

This repo contains the model codes and pretrained models for classification and semantic segmentation. You can refer to [
HRNet-Semantic-Segmentation-pytorch-v1.1](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/pytorch-v1.1) for train and test the models locally. 

# Pretrained models

DDRNet_23_slim on ImageNet(top-1:):

DDRNet_23 on ImageNet(top-1:):

DDRNet_39 on ImageNet(top-1:):

DDRNet_23_slim on Cityscapes(val mIoU:):

DDRNet_23 on Cityscapes(val mIoU:):

## Citation
If you find this repo is useful for your research, Please consider citing our paper:

```
@article{hong2021deep,
  title={Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation of Road Scenes},
  author={Hong, Yuanduo and Pan, Huihui and Sun, Weichao and Jia, Yisong and others},
  journal={arXiv preprint arXiv:2101.06085},
  year={2021}
}
```
