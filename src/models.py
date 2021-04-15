import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import segmentation_models_pytorch as smp

class ModelSM:
    def __init__(self, model_name, backbone, classes, encoder_weights=None, depth=5):
        self.model_name      = model_name.lower()
        self.backbone        = backbone
        self.classes         = classes
        self.encoder_weights = encoder_weights
        self.depth           = depth

        # define network parameters
        self.num_classes = 1 if len(self.classes) == 1 else (len(self.classes) + 1)
        self.activation = 'sigmoid' if self.num_classes == 1 else 'softmax'

        # Status of the model
        # if self.weights is None:
        #     self.is_compiled = False
        # else:
        #     self.is_compiled = True
        #     self._create_model()
    
    def get_num_classes(self):
        return self.num_classes
    
    def get_preprocess_input_fn(self):
        return smp.encoders.get_preprocessing_fn(self.backbone, pretrained='imagenet')
    
    def _create_model(self):
        if self.model_name=='unet':
            self.model = smp.Unet(
                encoder_name=self.backbone,
                encoder_depth=self.depth,
                encoder_weights=self.encoder_weights,
                decoder_use_batchnorm=True,
                decoder_attention_type='scse',
                classes=self.num_classes,
                activation=self.activation
            )
        elif self.model_name=='fpn':
            self.model = smp.FPN(
                encoder_name=self.backbone,
                encoder_depth=self.depth,
                encoder_weights=self.encoder_weights,
                decoder_dropout=0.2,
                upsampling=4,
                classes=self.num_classes,
                activation=self.activation
            )

    def create_model(self, learning_rate):
        """
        TODO: Add parameters to make custom model
        """
        
        self._create_model()

        # Define optimizer
        self.optim = torch.optim.Adam([dict(lr=learning_rate, params=self.model.parameters()),])
        
        if self.num_classes>1:
            dice_loss = smp.losses.DiceLoss(mode='binary')
            focal_loss = smp.losses.FocalLoss(mode='binary')
        else:
            dice_loss = smp.losses.DiceLoss(mode='multiclass')
            focal_loss = smp.losses.FocalLoss(mode='multiclass')
        
        # total_loss = lambda pr, gt: dice_loss(pr,gt) + focal_loss(pr,gt)
        total_loss = dice_loss
        total_loss.__name__ = 'dice_loss'
        self.loss = total_loss
        self.metrics = [smp.utils.metrics.IoU(threshold=0.5), smp.utils.metrics.Fscore(threshold=0.5)]

        # Model is compiled
        self.is_compiled = True
    
    def train_model(self, device, train_loader, valid_loader, epochs, output_file='./best_model.pth', args={}):
        """Train the model
        """
        if not self.is_compiled:
            raise Exception("Create model before fitting")
        
        if args['parallel']:
            rank = args['nr'] * args['gpus'] + args['gpu_id']
            dist.init_process_group(backend='nccl', init_method='env://', world_size=args['world_size'], rank=rank)
            torch.manual_seed(0)
            torch.cuda.set_device(args['gpu_id'])
            self.model.cuda(args['gpu_id'])
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args['gpu_id']], find_unused_parameters=True)

        # Create epoch runners - it is a simple loop of iterating over dataloader`s samples
        train_epoch = smp.utils.train.TrainEpoch(
            self.model,
            loss=self.loss,
            metrics=self.metrics,
            optimizer=self.optim,
            device=device,
            verbose=True,
        )
        valid_epoch = smp.utils.train.ValidEpoch(
            self.model,
            loss=self.loss,
            metrics=self.metrics,
            device=device,
            verbose=True,
        )

        # Train model
        max_score = 0
        all_epoch_train_logs = []
        all_epoch_valid_logs = []
        total_time_taken = 0.0
        for i in range(0, epochs):
            print('\nEpoch: {}'.format(i))
            _t0 = time.time()
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)
            total_time_taken += time.time() - _t0
            
            all_epoch_train_logs.append(train_logs)
            all_epoch_valid_logs.append(valid_logs)
            
            # TODO: Add more callbacks
            if max_score < valid_logs['iou_score'] + valid_logs['fscore']:
                max_score = valid_logs['iou_score']
                torch.save(self.model, output_file)

            if i == 25:
                self.optim.param_groups[0]['lr'] = 1e-5
                print('Decrease decoder learning rate to 1e-5!')
        
        return {
            'train': all_epoch_train_logs,
            'valid': all_epoch_valid_logs,
            'time_taken': total_time_taken
        }
    
    def predict(self, x):
        # Predict
        return self.model.predict(x)
