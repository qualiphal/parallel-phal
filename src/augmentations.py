import albumentations as A

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

class Augmentor:
    """
    img_size : (height, width)
    """
    def __init__(self, img_size):
        self.img_size = img_size
        self.width = self.img_size[1]
        self.height = self.img_size[0]
    
    def _round_clip_0_1(self, x, **kwargs):
        return x.round().clip(0, 1)

    # define heavy augmentations
    def get_training_augmentation(self):
        train_transform = [

            A.HorizontalFlip(p=0.5),

            A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

            A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
            A.RandomCrop(height=320, width=320, always_apply=True),

            A.IAAAdditiveGaussianNoise(p=0.2),
            A.IAAPerspective(p=0.5),

            # A.OneOf(
            #     [
            #         A.CLAHE(p=1),
            #         A.RandomBrightness(p=1),
            #         A.RandomGamma(p=1),
            #     ],
            #     p=0.9,
            # ),

            A.OneOf(
                [
                    A.IAASharpen(p=1),
                    A.Blur(blur_limit=3, p=1),
                    A.MotionBlur(blur_limit=3, p=1),
                ],
                p=0.9,
            ),

            # A.OneOf(
            #     [
            #         A.RandomContrast(p=1),
            #         A.HueSaturationValue(p=1),
            #     ],
            #     p=0.9,
            # ),
            # A.Lambda(mask=self._round_clip_0_1)
        ]
        return A.Compose(train_transform)

    def get_validation_augmentation(self):
        """Add paddings to make image shape divisible by 32"""
        test_transform = [
            A.PadIfNeeded(384, 480)
        ]
        return A.Compose(test_transform)

    def get_preprocessing(self, preprocessing_fn):
        """Construct preprocessing transform
        
        Args:
            preprocessing_fn (callbale): data normalization function 
                (can be specific for each pretrained neural network)
        Return:
            transform: albumentations.Compose
        
        """
        
        _transform = [
            A.Lambda(image=preprocessing_fn),
            A.Lambda(image=to_tensor, mask=to_tensor)
        ]
        return A.Compose(_transform)
