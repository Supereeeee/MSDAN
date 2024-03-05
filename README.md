# Efficient feature extraction network for image super-resolution(EFEN)
Yinggan Tang, Quanwei Hu, Chunning BU

## Environment

[BasicSR >= 1.4.2]
[PyTorch >= 1.13.0]
[Torchvision >= 0.14.0]
[Cuda >= 11.7]

### Installation
```
pip install -r requirements.txt
python setup.py develop
```

## How To Test
· Refer to ./options/test for the configuration file of the model to be tested, and prepare the testing data and pretrained model.  
· The pretrained models are available at [Google Drive] or [Baidu Netdisk]. Place the pretrained models in ./experiments/pretrained_models/  
· Then run the follwing codes (taking EFENx4.pth as an example):  

```
python basicsr/test.py -opt options/test/test_EFEN_x4.yml
```
The testing results will be saved in the ./results folder.

## How To Train
· Refer to ./options/train for the configuration file of the model to train.  
· Preparation of training data can refer to this page. All datasets can be downloaded at the official website.  
· Note that the default training dataset is based on lmdb, refer to [docs in BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md) to learn how to generate the training datasets.  
· The training command is like  
```
python basicsr/train.py -opt options/train/train_EFEN_x4.yml
```
For more training commands and details, please check the docs in [BasicSR](https://github.com/XPixelGroup/BasicSR)  

## Results
The inference results on benchmark datasets are available at [Google Drive] or [Baidu Netdisk].

## Acknowledgement
This code is based on [BasicSR](https://github.com/XPixelGroup/BasicSR) toolbox. Thanks for the awesome work.

## Contact
If you have any question, please email 1051823707@qq.com.
