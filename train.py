import warnings
warnings.filterwarnings('ignore')
import cv2
import numpy as np
import segmentation_models as sm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import keras
import argparse
from dataset import Dataset
from dataloader import Dataloader
from helpers import *

parser = argparse.ArgumentParser()
parser.add_argument('--backbone', help='Choose the backbone model, efficientnetb5 or mobilenetv2', type=str, required=True)
parser.add_argument('--dataset_path', help='Provide the dataset path', type=str, default='./data')
parser.add_argument('--epochs', help='Choose the number of epochs of the training', type=int, default=100)
parser.add_argument('--learning_rate', help='Choose the learning rate', type=int, default=0.001)
parser.add_argument('--batch_size', help='Choose the size of the training batches', type=int, default=5)
parser.add_argument('--patience', help='Choose the number of epochs that produced validation loss with no improvement after which training will be stopped ', type=int, default=15)

args = parser.parse_args()

#Training and validation folders (need to point to a folder with the following folder names)
x_train_dir = os.path.join(args.dataset_path, 'train')
y_train_dir = os.path.join(args.dataset_path, 'train_annotated')
x_valid_dir = os.path.join(args.dataset_path, 'validation')
y_valid_dir = os.path.join(args.dataset_path, 'validation_annotated')

if not os.path.exists(x_train_dir):
    raise ValueError('The dataset path \'{dataset_path}\' does not contain a folder named train.'.format(dataset_path=args.dataset_path))

if not os.path.exists(y_train_dir):
    raise ValueError('The dataset path \'{dataset_path}\' does not contain a folder named train_annotated.'.format(dataset_path=args.dataset_path))

if not os.path.exists(x_valid_dir):
    raise ValueError('The dataset path \'{dataset_path}\' does not contain a folder named validation.'.format(dataset_path=args.dataset_path))

if not os.path.exists(y_valid_dir):
    raise ValueError('The dataset path \'{dataset_path}\' does not contain a folder named validation_annotated.'.format(dataset_path=args.dataset_path))


# Build the model
backbones = ['efficientnetb5','mobilenetv2','efficientnetb3']

if(args.backbone not in backbones):
	raise ValueError('The backbone \'{backbone}\' does not exist. Choose mobilenetv2 or efficientnetb5'.format(backbone=args.backbone))

model = sm.Unet(args.backbone,input_shape=(480, 320, 3) ,classes=1, activation='sigmoid',decoder_filters=(128, 64, 32, 16,8))

preprocess_input = sm.get_preprocessing(args.backbone)

optimizer = keras.optimizers.Adam(args.learning_rate)

model.compile(optimizer, sm.losses.binary_focal_dice_loss, [sm.metrics.IOUScore(threshold=0.5)])


# Dataset for training images
train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    classes=['Avalanche'], 
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)

# Dataset for validation images
valid_dataset = Dataset(
    x_valid_dir, 
    y_valid_dir, 
    classes=['Avalanche'], 
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)

# Define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
    keras.callbacks.ModelCheckpoint(args.backbone+'.h5', save_weights_only=True, save_best_only=True, mode='min'),
    keras.callbacks.EarlyStopping(monitor='val_loss',patience=args.patience),
    keras.callbacks.ReduceLROnPlateau(),
]

# Define the data generators
train_dataloader = Dataloader(train_dataset, batch_size=args.batch_size)
valid_dataloader = Dataloader(valid_dataset, batch_size=1)

if(train_dataloader[0][0].shape != (args.batch_size, 480, 320,3)):
	raise ValueError('The training images provided in path \'{dataset_path}\' do not have (480,320) sizes.'.format(dataset_path=args.dataset_path))

if(valid_dataloader[0][1].shape == (args.batch_size, 480, 320,1)):
	raise ValueError('The validation images provided in path \'{dataset_path}\' do not have (480,320) sizes.'.format(dataset_path=args.dataset_path))

# Train model
history = model.fit_generator(
    train_dataloader, 
    steps_per_epoch=len(train_dataloader), 
    epochs=args.epochs, 
    callbacks=callbacks, 
    validation_data=valid_dataloader, 
    validation_steps=len(valid_dataloader),
    workers=os.cpu_count()
)
