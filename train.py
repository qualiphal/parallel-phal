import os
import numpy as np
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from src.models import ModelSM
from src.datasets import LemonDatasetCOCO
from src.augmentations import Augmentor
from utils.paths import Path

# Config parameters
DEVICE     = 'cuda'
MODEL_NAME = 'unet'
BACKBONE   = 'efficientnet-b3'
ENCODER_WEIGHTS = 'imagenet' # None -> random initialization
BATCH_SIZE = 4
CLASSES    = ['illness','gangrene']
LR         = 0.0001
EPOCHS     = 100
IMG_SIZE   = (256,256)
OUTPUT_FILE = './best_model_unet2.h5'
LIMIT_IMAGES = 100

# Group model parameters
model = ModelSM(
    model_name=MODEL_NAME,
    backbone=BACKBONE,
    classes=CLASSES,
    encoder_weights=ENCODER_WEIGHTS,
    depth=5
)

# Prepare augmentation
augmentor = Augmentor(img_size=IMG_SIZE)

# Create train and val dataloader
train_dataset = LemonDatasetCOCO(
    images_dir=Path.get_x_train_dir(),
    annot_file=Path.get_y_train_file(),
    img_size=IMG_SIZE,
    classes=CLASSES,
    augmentation=augmentor.get_training_augmentation(),
    preprocessing=augmentor.get_preprocessing(model.get_preprocess_input_fn()),
    limit_images=LIMIT_IMAGES
)
valid_dataset = LemonDatasetCOCO(
    images_dir=Path.get_x_val_dir(),
    annot_file=Path.get_y_val_file(),
    img_size=IMG_SIZE,
    classes=CLASSES,
    augmentation=augmentor.get_validation_augmentation(),
    preprocessing=augmentor.get_preprocessing(model.get_preprocess_input_fn()),
    limit_images=LIMIT_IMAGES
)

print("Training dataset length:", len(train_dataset))
print("Validation dataset length:", len(valid_dataset))

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2)

# check shapes for errors
# dataloader element, for keras: (BATCH_SIZE, *IMG_SIZE, 3) ;; for torch: (BATCH_SIZE, 3, *IMG_SIZE)
# dataloader element, for keras and torch, (BATCH_SIZE, *IMG_SIZE, model.get_num_classes())

# Create model
model.create_model(learning_rate=LR)

# Train model
model.train_model(
    device=DEVICE,
    train_loader=train_loader,
    valid_loader=valid_loader,
    epochs=EPOCHS
)


print(model.model.summary())

# Define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
    keras.callbacks.ModelCheckpoint(OUTPUT_FILE, save_weights_only=True, save_best_only=True, save_freq=len(train_dataloader)*5, mode='min'),
    keras.callbacks.ReduceLROnPlateau(),
    keras.callbacks.TensorBoard()
]

# Train model
history = model.fit_generator(
    train_generator=train_dataloader,
    valid_generator=valid_dataloader,
    epochs=EPOCHS,
    callbacks=callbacks
)

import matplotlib.pyplot as plt
# Plot training & validation iou_score values
plt.figure(figsize=(30, 5))
plt.subplot(121)
plt.plot(history.history['iou_score'])
plt.plot(history.history['val_iou_score'])
plt.title('Model iou_score')
plt.ylabel('iou_score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()