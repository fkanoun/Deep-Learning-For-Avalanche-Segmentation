import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
import warnings
warnings.filterwarnings('ignore')

def visualize(**images):
    """ Plot images next to each other  """
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
     
def denormalize(x):
    """ Scale image for correct plot   """
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

def get_training_augmentation():
    """ Apply random transformations at each epoch  """
    train_transform = [

        A.HorizontalFlip(p=0.5), #Rotation on y axes

        A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=10, shift_limit=0.05, p=0.5, border_mode=0),

        #A.RandomSizedCrop((320 , 320 ), height=480, width=320), #Crop

        A.IAAAdditiveGaussianNoise(p=0.2),

        A.IAAPerspective(p=0.5),

        A.OneOf([A.CLAHE(p=1),A.RandomBrightness(p=1),A.RandomGamma(p=1)],p=0.5),

        A.OneOf([A.RandomContrast(p=1),A.HueSaturationValue(p=1)],p=0.5),
        
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)

def get_validation_augmentation():
    """ Add paddings to make image shape 480,320 """
    test_transform = [
        A.PadIfNeeded(480, 320)
    ]
    return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """ Construct appropriate preprocessing transform for each backbone """
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)