import warnings
warnings.filterwarnings('ignore')
import cv2
import numpy as np
import segmentation_models as sm
from helpers import visualize, denormalize
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import keras
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--backbone', help='Choose the backbone model, efficientnetb3 or mobilenetv2', type=str, default='mobilenetv2')
parser.add_argument('--pretrained', help='Choose whether to use the pretrained model or the new one', type=str2bool, default=True)
parser.add_argument('--image_path', help='Provide the path of the image', type=str, default='test.jpg')
parser.add_argument('--timer', help='Provide the path of the image', type=str2bool, default=False)

args = parser.parse_args()

# check the image path
if not os.path.exists(args.image_path):
    raise ValueError('The path \'{image_path}\' does not exist the image file.'.format(image_path=args.image_path))

if (args.timer==True):
	time_start = time.clock()

#build the model
backbones = ['efficientnetb5','mobilenetv2']
if(args.backbone not in backbones):
	raise ValueError('The backbone \'{backbone}\' does not exist. Choose mobilenetv2 or efficientnetb5'.format(backbone=args.backbone))

print('Building the model...')
model = sm.Unet(args.backbone,input_shape=(480, 320, 3) ,classes=1, activation='sigmoid',decoder_filters=(128, 64, 32, 16,8))

#load the weights
print('Loading the weights...')
model_path = args.backbone
if(args.pretrained==True):
	model_path = model_path+'_pretrained' 

model_path = model_path+'.h5'
model.load_weights(model_path)

if(args.timer==True):
	time_elapsed0 = (time.clock() - time_start)
	print("****** Loading the model takes","%.2f" % time_elapsed0,'s *******')

#load the image
image = cv2.imread(args.image_path) #Read image
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) #Turn image to RGB after imread reads it in BGR
image = cv2.resize(image,(320,480))
image = np.expand_dims(image, axis=0) #Arrange the shape of the image
preprocess_input = sm.get_preprocessing(args.backbone) #Normalize features of the image
image = preprocess_input(image)

#infer the segmentation
print('Infering the avalanche...')
time_start = time.clock()
prediction = model.predict(image)

if(args.timer==True):
	time_elapsed = (time.clock() - time_start)
	print("****** Infering the avalanche takes","%.2f" % time_elapsed,'s *******')

if(args.timer==True):
	print('******  Total time',"%.2f" % (time_elapsed0+time_elapsed),'s ****** ')

#Visualize the results
pr_mask = prediction.round()
visualize(image=denormalize(image.squeeze()),Prediction_mask=pr_mask[...,0].squeeze(),)