# Style-Based Global Appearance Flow for Virtual Try-On (CVPR 2022)

## Requirements

- python 3.6
- pytorch 1.1.0 (as no third party libraries are required for this codebase, other version should also work, not yet tested)
- tensorboardX
- opencv
## Training ( `cd` to the train folder)

For VITON dataset, download the training data from [VITON_train](https://drive.google.com/file/d/1Uc0DTTkSfCPXDhd4CMx2TQlzlC6bDolK/view?usp=sharing) and put the folder `VITON_traindata` under the folder`train/datase` 

For perceptual loss computation, download  the vgg checkpoint from [VGG_Model](https://drive.google.com/file/d/1Mw24L52FfOT9xXm3I1GL8btn7vttsHd9/view?usp=sharing) and put `vgg19-dcbb9e9d.pth` under the folder `train/models`.

### Custom dataset

For other custom dataset, please generate the training data folder with the same structure as `VITON_traindata`.

More specifically, you need to prepare human parsing, pose (18 key points, saved in .json file) and densepose (a heatmap, different body region has different value).

For human parsing, you can use [Human parser](https://github.com/levindabhi/Self-Correction-Human-Parsing-for-ACGPN).

For pose, you can use [Human pose](https://github.com/Hzzone/pytorch-openpose).

For dense pose, you can use [Dense pose](https://github.com/facebookresearch/DensePose).


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
