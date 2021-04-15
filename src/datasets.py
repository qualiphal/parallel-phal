import os
import cv2
import numpy as np
from pycocotools.coco import COCO
import torch
from utils import cocoFunctions

class LemonDatasetCOCO(torch.utils.data.Dataset):
    """Lemon Dataset in COCO format made usable for torch dataloaders
    """
    CLASSES = ['image_quality','illness','gangrene','mould','blemish','dark_style_remains','artifact','condition','pedicel']

    def __init__(
        self,
        images_dir,
        annot_file,
        img_size,
        classes=None,
        augmentation=None,
        preprocessing=None
    ):
        # Initiate COCO API
        self.coco = COCO(annot_file)
        self.img_size = img_size

        # Load images in dict format using COCO
        self.ids = self.coco.getImgIds()
        self.images_dir = images_dir
        self.images_objs = [imgobj for imgobj in self.coco.imgs.values()]

        # Get class values
        self.classes = classes
        if self.classes is None:
            self.classes = [_cat['name'] for _cat in self.coco.loadCats(self.coco.getCatIds())]
        
        self.class_values = [self.coco.getCatIds(_cls)[0] for _cls in self.classes]

        # Get preprocessing and augmentation fn
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        # Read data
        image = cv2.imread(os.path.join(self.images_dir, os.path.basename(self.images_objs[i]['file_name'])))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask  = cocoFunctions.getNormalMask(self.coco, self.images_objs[i], self.classes)

        # Extract certain classes from mask
        masks = [(mask==v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype(np.float32)

        # Add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
        
        # Apply augmentations
        if self.augmentation is not None:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)
