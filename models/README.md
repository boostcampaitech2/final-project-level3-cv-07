# End-to-end Scene Text Spotting


## Prerequisites

This project using [Pytorch](https://pytorch.org/) - An open source machine learning framework.

Install from requirements.txt file 

```
cd models

pip install -r requirements.txt
```

## Dataset Structure

During training, we use [ICDAR 2015](https://rrc.cvc.uab.es/?ch=4&com=downloads)&[ICDAR2017](https://rrc.cvc.uab.es/?ch=8&com=downloads). In addition, we use [ICDAR_2017_valid](https://rrc.cvc.uab.es/?ch=8&com=downloads) for test our model.<br /><br />

The dataset should structure as follows:

```
[dataset root directory]
├── train_images
│   ├── img_1.jpg
│   ├── img_2.jpg
│   └── ...
├── train_gts
│   ├── gt_img_1.txt
│   ├── gt_img_2.txt
│   └── ...
├── test_images
|   ├── img_1.jpg
|   ├── img_2.jpg
|    └── ...
└── train_gts
    ├── gt_img_1.txt
    ├── gt_img_2.txt
    └── ...
```
Note: the [dataset root directory] should be placed in "config.json" file. <br /><br />
Sample of ground truth format:
```
x1,y1,x2,y2,x3,y3,x4,y4,script,transcription
```
## Wandb setting
```
wandb login
```
Type the above command.
If a browser link appears, click that, and copy and paste the API Key.
You can visualize the model's result.

## Training

Training the model by yourself
```
python train.py
```
Note: check the "config.json" file, which is used to adjust the training configuration.<br /><br />

## Inference
Test the model you trained
```
python eval.py
```
Test the model with checking the score
```
python test.py
```

If you want to change "your model, input images, output folder", check the eval.py file.

## Running Demo

Running the demo of our pre-trained [model](https://drive.google.com/file/d/1toEqT1LA-0ieY0ZFeKc6UWJOVvPXDtF1/view?usp=sharing)

```
python demo.py -m=model_best.pth.tar
```

## Acknowledgments

* https://github.com/ishin-pie/e2e-scene-text-spotting
* https://github.com/argman/EAST
* https://github.com/jiangxiluning/FOTS.PyTorch

