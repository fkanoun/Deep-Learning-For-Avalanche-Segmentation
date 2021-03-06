# Deep-Learning-For-Avalanche-Segmentation

## Installation
- Numpy `pip install numpy`
- Matplotlib `pip install matplotlib`
- Pillow `pip install Pillow`
- OpenCV `pip install opencv-python`
- Segmentation-Models `pip install -U --pre segmentation-models --user`
- Albumentations `pip install -U albumentations>=0.3.0 --user`
- Tensorflow `pip install tensorflow==1.3.0` 
- Keras `pip install keras==2.2.4`

**Note:** The working versions for this project are 1.3 for tensorflow and 2.2.4 for keras

***
## Files Setting
The files must satisfy the following structure:
```buildoutcfg
|-- train.py
|-- predict.py
|-- dataset.py
|-- dataloader.py
|-- helpers.py
|-- test.jpg
|-- efficientnetb5_pretrained.h5
|-- mobilenetv2_pretrained.h5
|-- data
|  |-- train
|  |-- train_annotated
|  |-- validation
|  |-- valid_annotated

```
**Note:** The folders of your data folder must satisfy the above structure. All images and masks should have (480,320) sizes.

***
## Training
This project contains complete codes for training and predicting. Possibility to train two different models, efficientnet or a lighter model mobilenet.
And you can perform a simple command as this to build a model on your dataset:
```buildoutcfg
python train.py --backbone mobilenetv2 --dataset DATASET_PATH 
```
The detailed command line parameters are as follows:
```buildoutcfg
usage: train.py [-h] --backbone BACKBONE [--dataset_path DATASET_PATH]
                [--epochs EPOCHS] [--learning_rate LEARNING_RATE]
                [--batch_size BATCH_SIZE] [--patience PATIENCE]

```
```buildoutcfg
optional arguments:
  -h, --help            Show this help message and exit
  --backbone BACKBONE   Choose the backbone model, efficientnetb5 or mobilenetv2
  --dataset_path DATASET_PATH 
                        Provide the dataset path
  --epochs EPOCHS       Choose the number of epochs of the training
  --learning_rate LEARNING_RATE 
                        Choose the learning rate
  --batch_size BATCH_SIZE
                        Choose the size of the training batches
  --patience PATIENCE   Choose the number of epochs that produced validation loss with no improvement after which training will be stopped
  

```

**Note:** If you don't give the parameter BACKBONE, the default backbone will be used. High computational resources are needed in order to train with efficientnetb5 backbone. 

***
## Predicting
You can get the avalanche segmentation from a mountain image using the pretrained models or your own trained models from above:
```buildoutcfg
python predict.py --backbone efficientnetb5 --pretrained true --timer false --image_path IMAGE_PATH
```
```buildoutcfg
optional arguments:
  -h, --help            Show this help message and exit
  --backbone BACKBONE   Choose the backbone model, efficientnetb5 or mobilenetv2
  --pretrained PRETRAINED
                        Use pretrained models or newly trained ones.
  --image_path IMAGE_PATH
                        Path to the mountain image
  --timer TIMER         Use timer or not.
  

``` 
**Note:** If no image is provided, the inference will be computed on the test.jpg image. Only the pretrained mobilenet backbone is available. If you want to test the efficientnetb5 model, please download it from this link (https://gofile.io/?c=825n0S) and place it in the main folder as specified in the files setting.

## Results

Original            |  GT   |  Result
:-------------------------:|:-------------------------:|:-------------------------:
![alt-text-3](https://github.com/fkanoun/Deep-Learning-For-Avalanche-Segmentation/blob/master/Images/1.png "Original")  |  ![alt-text-4](https://github.com/fkanoun/Deep-Learning-For-Avalanche-Segmentation/blob/master/Images/1_true.png "GT")  |   ![alt-text-5](https://github.com/fkanoun/Deep-Learning-For-Avalanche-Segmentation/blob/master/Images/1_pred.png "Result")
![alt-text-3](https://github.com/fkanoun/Deep-Learning-For-Avalanche-Segmentation/blob/master/Images/2.png "Original")  |  ![alt-text-4](https://github.com/fkanoun/Deep-Learning-For-Avalanche-Segmentation/blob/master/Images/2_true.png "GT")  |   ![alt-text-5](https://github.com/fkanoun/Deep-Learning-For-Avalanche-Segmentation/blob/master/Images/2_pred.png "Result")
![alt-text-3](https://github.com/fkanoun/Deep-Learning-For-Avalanche-Segmentation/blob/master/Images/3.png "Original")  |  ![alt-text-4](https://github.com/fkanoun/Deep-Learning-For-Avalanche-Segmentation/blob/master/Images/3_true.png "GT")  |   ![alt-text-5](https://github.com/fkanoun/Deep-Learning-For-Avalanche-Segmentation/blob/master/Images/3_pred.png "Result")
![alt-text-3](https://github.com/fkanoun/Deep-Learning-For-Avalanche-Segmentation/blob/master/Images/4.png "Original")  |  ![alt-text-4](https://github.com/fkanoun/Deep-Learning-For-Avalanche-Segmentation/blob/master/Images/4_true.png "GT")  |   ![alt-text-5](https://github.com/fkanoun/Deep-Learning-For-Avalanche-Segmentation/blob/master/Images/4_pred.png "Result")
