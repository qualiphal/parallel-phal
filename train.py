import os
import torch
import pickle
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

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
EPOCHS     = 5
IMG_SIZE   = (256,256)
LIMIT_IMAGES = 32
OUTPUT_FILE = './best_model.pt'
EXP_NAME    = './logs/unet_exp0.pkl'
PARALLEL   = False

# Parallel configuration
args = {
    'nodes':1,
    'gpus':1,
    'nr':0,
    'gpu_id':0,
    'parallel':PARALLEL
}
args['world_size'] = args['gpus'] * args['nodes']
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '8888'

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

if PARALLEL:
    # Sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args['world_size'],
        rank=args['nr'] * args['gpus'] + args['gpu_id']
    )
    # Create train data loader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True, sampler=train_sampler)
else:
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)

# Create valid data loader
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2)

# check shapes for errors
# dataloader element, for keras: (BATCH_SIZE, *IMG_SIZE, 3) ;; for torch: (BATCH_SIZE, 3, *IMG_SIZE)
# dataloader element, for keras and torch, (BATCH_SIZE, *IMG_SIZE, model.get_num_classes())

# Create model
model.create_model(learning_rate=LR)

# Train model
logs = model.train_model(
    device=DEVICE,
    train_loader=train_loader,
    valid_loader=valid_loader,
    epochs=EPOCHS,
    output_file=OUTPUT_FILE,
    args=args
)

# Write results
with open(EXP_NAME, 'wb') as handle:
    pickle.dump(logs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(logs['time_taken'])
