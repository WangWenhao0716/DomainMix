# DomainMix
[BMVC2021] The official implementation of "DomainMix: Learning Generalizable Person Re-Identification Without Human Annotations"

[[paper](https://arxiv.org/pdf/2011.11953.pdf)] [[demo]()] [[Chinese blog]()]


DomainMix works fine on both [PaddlePaddle](https://www.paddlepaddle.org.cn/) and [PyTorch](https://pytorch.org/).

Framework:
<div align=center><img src="https://github.com/WangWenhao0716/DomainMix/blob/main/framework.png" width="80%"/></div>

## Requirement
* Python 3.7
* Pytorch 1.7.0
* sklearn 0.23.2
* PIL 5.4.1
* Numpy 1.19.4
* Torchvision 0.8.1

## Reproduction Environment
* Test our models: 1 Tesla V100 GPU.
* Train new models: 4 Telsa V100 GPUs.
* Note that the required for GPU is not very strict, and 6G memory per GPU is minimum.

## Preparation
1. Dataset

We evaluate our algorithm on [**RandPerson**](https://dl.acm.org/doi/abs/10.1145/3394171.3413815), [**Market-1501**](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf), [**CUHK03-NP**](https://github.com/zhunzhong07/person-re-ranking/tree/master/CUHK03-NP) and [**MSMT17**](https://arxiv.org/abs/1711.08565). You should download them by yourselves and prepare the directory structure like this:

```
*DATA_PATH
      *data
         *randperson_subset
             *randperson_subset
                 ...
         *market1501
             *Market-1501-v15.09.15
                 *bounding_box_test
                 ...
         *cuhk03_np
             *detected
             *labeled
         *msmt17
             *MSMT17_V1
                 *test
                 *train
                 ...
```

2. Pretrained Models

We use ResNet-50 and [IBN-ResNet-50](https://arxiv.org/abs/1807.09441) as backbones. The pretrained models for ResNet-50 will be downloaded automatically. When training with the backbone of IBN-ResNet-50, you should download the pretrained models from [here](https://drive.google.com/drive/folders/1thS2B8UOSBi_cJX6zRy6YYRwz_nVFI_S), and save it like this:
```
*DATA_PATH
      *logs
         *pretrained
             resnet50_ibn_a.pth.tar
```

3. Our Trained Models

We provide our trained models as follows. They should be saved in `./logs/trained`

Market1501:

[**DomainMix(43.5% mAP)**](https://drive.google.com/file/d/1-20fEHgTNi66OBm2i_UEDFj8gYd7u19p/view?usp=sharing)    [**DomainMix-IBN(45.7% mAP)**](https://drive.google.com/file/d/1-5NrcubD3SFgpWbXstIB8WX-CTrxOJMf/view?usp=sharing)


CUHK03-NP:

[**DomainMix(16.7% mAP)**](https://drive.google.com/file/d/1JPlslEzafMjI7bxoq9C34p8Be56VrFvk/view?usp=sharing)    [**DomainMix-IBN(18.3% mAP)**](https://drive.google.com/file/d/1-0p6X6QHerJ4CvhcJEe0V9peyhl1Vvtj/view?usp=sharing)


MSMT17:

[**DomainMix(9.3% mAP)**](https://drive.google.com/file/d/1-7vQG8os0bb-beNQHeZ_4VH1jgUt4EpQ/view?usp=sharing)    [**DomainMix-IBN(12.1% mAP)**](https://drive.google.com/file/d/1-BFX0MUT-_UPYoEEJ04B7hL0bbsQ-2W9/view?usp=sharing)


## Train

We use RandPerson+MSMT->Market as an example, other DG tasks will follow similar pipelines.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
-dsy randperson_subset -dre msmt17 -dun market1501 \
-a resnet50 --margin 0.0 --num-instances 4 -b 64 -j 4 --warmup-step 5 \
--lr 0.00035 --milestones 10 15 30 40 50 --iters 2000 \
--epochs 60 --eval-step 1 --logs-dir logs/randperson_subsetmsTOm/domainmix
```


## Test

We use RandPerson+MSMT->Market as an example, other DG tasks will follow similar pipelines.
```
CUDA_VISIBLE_DEVICES=0 python test.py -b 256 -j 8 --dataset-target market1501 -a resnet50 \
--resume logs/trained/model_best_435.pth.tar
```

## Acknowledgement

Some parts of our code are from [MMT](https://github.com/yxgeee/MMT) and [SpCL](https://github.com/yxgeee/SpCL). Thanks Yixiao Ge for her contribution.

## Citation
If you find this code useful for your research, please cite our paper
```
@inproceedings{wang2021domainmix,
  title={DomainMix: Learning Generalizable Person Re-Identification Without Human Annotations},
  author={Wenhao Wang and Shengcai Liao and Fang Zhao and Kangkang Cui and Ling Shao},
  booktitle={British Machine Vision Conference},
  year={2021}
}
```

