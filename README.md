# Style-Based Global Appearance Flow for Virtual Try-On (CVPR 2022)
### [Project Page](https://github.com/SenHe/Flow-Style-VTON/) | [Paper](https://arxiv.org/pdf/2204.01046.pdf) | [Video](https://www.youtube.com/watch?v=Og7IDzQJwZQ) | [Poster](https://github.com/SenHe/Flow-Style-VTON/blob/main/poster.pdf) | [Supplementary Material](https://github.com/SenHe/Flow-Style-VTON/blob/main/supp.pdf)
<div align="center">
<img src=gif_detail.gif width="800">
</div>

## Requirements

- python 3.6.13
- torch 1.1.0 (as no third party libraries are required in this codebase, other versions should work, not yet tested)
- torchvision 0.3.0
- tensorboardX
- opencv

## Inference (`cd` to test folder)

Download the testing data from [here](https://drive.google.com/file/d/1Y7uV0gomwWyxCvvH8TIbY7D9cTAUy6om/view).

If you want to test on the augmented testing images, download the images from [here](https://drive.google.com/drive/folders/1tUjnPW2_HfC7tpRBYG9xh3f3DfGL2CSk?usp=sharing) and put it to the testing data folder downloaded above, change the [image folder here](https://github.com/SenHe/Flow-Style-VTON/blob/dc3ddc5b16b1905c69acba8dfbe70ec66dcb91ec/test/data/aligned_dataset_test.py#L16) to `_ma_img`.

Download pretrained checkpoints from [here](https://drive.google.com/drive/folders/1hunG-84GOSq-qviJRvkXeSMFgnItOTTU?usp=sharing).
The checkpoint trained without augmentation is better for testing set in the VITON. But the checkpoint trained with augmentation is more robust for in-the-wild images.
```
python test.py --name demo --resize_or_crop None --batchSize 1 --gpu_ids 0 --warp_checkpoint your_path_to_the_down_loaded_ckp/PFAFN_warp_epoch_101.pth --gen_checkpoint your_path_to_the_down_loaded_ckp/PFAFN_gen_epoch_101.pth --dataroot 'your_path_to_the_downloaded_test_data'
```

## Test
[FID](https://github.com/mseitzer/pytorch-fid) and [SSIM](https://github.com/Po-Hsun-Su/pytorch-ssim)

## Training ( `cd` to the train folder)

For VITON dataset, download the training data from [VITON_train](https://drive.google.com/file/d/1Uc0DTTkSfCPXDhd4CMx2TQlzlC6bDolK/view?usp=sharing) and put the folder `VITON_traindata` under the folder`train/dataset` 

For perceptual loss computation, download  the vgg checkpoint from [VGG_Model](https://drive.google.com/file/d/1Mw24L52FfOT9xXm3I1GL8btn7vttsHd9/view?usp=sharing) and put `vgg19-dcbb9e9d.pth` under the folder `train/models`.

### Custom dataset

For other custom dataset, please generate the training data folder with the same structure as `VITON_traindata`.

More specifically, you need to prepare human parsing, pose (18 key points, saved in .json file) and densepose (a heatmap, different body region has different value).

For human parsing, you can use [Human parser](https://github.com/levindabhi/Self-Correction-Human-Parsing-for-ACGPN).

For pose, you can use [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose).

For dense pose, you can use [Dense pose](https://github.com/facebookresearch/DensePose).

__Note:__ To train the model on your own dataset, you may need to decrease the [learning rate](https://github.com/SenHe/Flow-Style-VTON/blob/a6a17405fe4d864ef1dd9d078fd9f2dd23b6ec35/train/options/train_options.py#L25) to 0.00001.


__Note:__ if you want to train with augmentation, check [here](https://github.com/SenHe/Flow-Style-VTON/blob/785a00fa4ce68fa0cee9f8247f1dc2d35e946842/train/train_PBAFN_stage1_fs.py#L21), [here](https://github.com/SenHe/Flow-Style-VTON/blob/785a00fa4ce68fa0cee9f8247f1dc2d35e946842/train/train_PBAFN_e2e_fs.py#L21), [here](https://github.com/SenHe/Flow-Style-VTON/blob/785a00fa4ce68fa0cee9f8247f1dc2d35e946842/train/train_PFAFN_stage1_fs.py#L23) and [here](https://github.com/SenHe/Flow-Style-VTON/blob/785a00fa4ce68fa0cee9f8247f1dc2d35e946842/train/train_PFAFN_e2e_fs.py#L22) in the code.


### Stage 1: Parser-Based Appearance Flow Style
```
sh scripts/train_PBAFN_stage1_fs.sh
```
### Stage 2: Parser-Based Generator
```
sh scripts/train_PBAFN_e2e_fs.sh
```

### Stage 3: Parser-Free Appearance Flow Style
```
sh scripts/train_PFAFN_stage1_fs.sh
```

### Stage 4: Parser-Free Generator
```
sh scripts/train_PFAFN_e2e_fs.sh
```


## Reference

If you find this repo helpful, please consider citing:

```
@inproceedings{he2022fs_vton,
  title={Style-Based Global Appearance Flow for Virtual Try-On},
  author={He, Sen and Song, Yi-Zhe and Xiang, Tao},
  booktitle={CVPR},
  year={2022}
}
```

## Acknowledgements

This repository is based on [PF-AFN](https://github.com/geyuying/PF-AFN), where we replaced the [tensor correlation](https://github.com/lmb-freiburg/flownet2) based flow estimation with our proposed style-based flow estimation.
