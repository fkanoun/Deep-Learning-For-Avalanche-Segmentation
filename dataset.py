import os
import cv2
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class Dataset:
    """ Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation : data transfromation pipeline 
        preprocessing : data preprocessing 
    
    """
    
    CLASSES = ['Avalanche']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids if ((image_id!='.DS_Store') and (image_id!='.ipynb_checkpoints'))] #images file paths
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids if ((image_id!='.DS_Store') and (image_id!='.ipynb_checkpoints'))] #masks file paths
        
        # convert str names to class numbers of masks
        self.class_values = [self.CLASSES.index(cls)+1 for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i]) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Turn image to RGB after imread reads it in BGR
        #print(self.images_fps[i])
        
        #mask contains for each pixel the classlabel it represents
        mask = cv2.imread(self.masks_fps[i], 0)
        # extract a one hot encoded mask for each class
        masks = [(mask == v) for v in self.class_values]
        
        
        #stack the masks along the third axis
        mask = np.stack(masks, axis=-1).astype('float')
        

        # add background mask (all what remains is put there)
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.images_fps) 