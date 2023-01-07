# The official implementation of "Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation of Road Scenes"
 
![avatar](./figs/performance.png)

**Achieve state-of-the-art trade-off between accuracy and speed on cityscapes and camvid, without using inference acceleration (like tensorRT) and extra data (like Mapillary)!** 

![avatar](./figs/DDRNet_seg.png)
The overall architecture of our methods.

![avatar](./figs/DAPPM.png)
The details of "Deep Aggregation Pyramid Pooling Module(DAPPM)".

## Usage

Currently, this repo contains the model codes and pretrained models for classification and semantic segmentation. Our models are trained using this code base
[HRNet-Semantic-Segmentation-pytorch-v1.1](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/pytorch-v1.1).

For training DDRNet, you can refer to [DDRNet.pytorch](https://github.com/chenjun2hao/DDRNet.pytorch), [Segmentation-Pytorch](https://github.com/Deeachain/Segmentation-Pytorch), [semantic-segmentation](https://github.com/sithu31296/semantic-segmentation). **PaddleSeg** has supported DDRNet-23 now, which achieves **79.85** mIoU. Thanks for their works!

## Notice

There are some basic training tricks you should employ to reproduce our results including class balance sample, ohem, crop size of 1024x1024. More details can be found in the [paper](https://arxiv.org/abs/2101.06085). And there is usually some variation with Cityscapes val results of the same model, maybe about 1% mIoU.

Keep "align_corners=False" in all places if you want to use our pretrained models for evaluation directly.



## Pretrained models

### ImageNet

DDRNet_23_slim(top-1 error:29.8): [googledrive](https://drive.google.com/file/d/1mg5tMX7TJ9ZVcAiGSB4PEihPtrJyalB4/view?usp=sharing)

DDRNet_23_slim using timm library, maybe helpful to train on own datasets(top-1 error:26.3, trained with a batch size of 256, warmup, cosine learning rate, 300 epoches and label smoothing): [googledrive](https://drive.google.com/file/d/17sgZ8mRJFhsItmdTrifI1rloVq5K1WiC/view?usp=sharing)

DDRNet_23(top-1 error:24.1): [googledrive](https://drive.google.com/file/d/1VoUsERBeuCaiuQJufu8PqpKKtGvCTdug/view?usp=sharing)

DDRNet_39(top-1 error:22.7): [googledrive](https://drive.google.com/file/d/122CMx6DZBaRRf-dOHYwuDY9vG0_UQ10i/view?usp=sharing)

### Cityscapes

DDRNet_23_slim(val mIoU:77.8): [googledrive](https://drive.google.com/file/d/1d_K3Af5fKHYwxSo8HkxpnhiekhwovmiP/view?usp=sharing)

DDRNet_23(val mIoU:79.5): [googledrive](https://drive.google.com/file/d/16viDZhbmuc3y7OSsUo2vhA7V6kYO0KX6/view?usp=sharing)

### CamVid

Dataset can be downloaded from the [link](https://paddleseg.bj.bcebos.com/dataset/camvid.tar).

DDRNet_23_slim: [googledrive](https://drive.google.com/file/d/1sh71nLdFKq1l89X3xyVO2J0d_3qBZui8/view?usp=sharing)

### [Comma10K](https://github.com/commaai/comma10k)
 Methods | Val loss | FPS  
:--:|:--:|:--:
 UNet-EfficientNetB0     | 0.0495 | 35.6 |   
 UNet-EfficientNetB4  | 0.0462 | 18.0  |  
 STDC1-Seg   | 0.0482 | 92.0  | 
 STDC2-Seg   | 0.0448 | 73.0  |
 DDRNet_23_slim   | 0.0448 | 166.8  |
 DDRNet_23   | 0.0433 | 62.7  |
 DDRNet_39   | 0.0428 | 36.3  |
 
 Please refer to [comma10k-baseline](https://github.com/YassineYousfi/comma10k-baseline) for train and test details. The FPS is tested with the method of our paper under the same conditions.

## Results on Cityscapes server

DDRNet_23_slim: [77.4](https://www.cityscapes-dataset.com/anonymous-results/?id=552a0548931fb49759bde6216f8472f60c470f768ac78b4cd08bf30a3a161e82)

DDRNet_23: [79.4](https://www.cityscapes-dataset.com/anonymous-results/?id=5766a6aff8efa27239e2f1d1085052cdb0a2351a66ef00d1610c9ea226e6770b)

DDRNet_39: [80.4](https://www.cityscapes-dataset.com/anonymous-results/?id=c9a859907b83426a71dcdcb08a7c0ad5b69111a45e61e3fdef5df1ddc680268c) [81.9](https://www.cityscapes-dataset.com/anonymous-results/?id=594e60787c8af8203cd37e5094c764a93b5a0c35e1e699d89ce4a64cb9da447b)(multi-scale and flip)

DDRNet_39 1.5x: [82.4](https://www.cityscapes-dataset.com/anonymous-results/?id=3515d66c1dc86c6daf42800c85a2937205658c6a8e5880904f350d8af234db01)(multi-scale and flip)

## Test Speed
Evaluate the inference speed on Cityscapes dataset.
```
python3 DDRNet_23_slim_eval_speed.py
```
DDRNet-23-slim can achieve above 130fps by using the [tool](https://github.com/NVIDIA-AI-IOT/torch2trt).

## Citation
If you find this repo is useful for your research, Please consider citing our paper:

```
@article{hong2021deep,
  title={Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation of Road Scenes},
  author={Hong, Yuanduo and Pan, Huihui and Sun, Weichao and Jia, Yisong},
  journal={arXiv preprint arXiv:2101.06085},
  year={2021}
}

@article{pan2022deep,
  title={Deep Dual-Resolution Networks for Real-Time and Accurate Semantic Segmentation of Traffic Scenes},
  author={Pan, Huihui and Hong, Yuanduo and Sun, Weichao and Jia, Yisong},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2022},
  publisher={IEEE}
}
```
