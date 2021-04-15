import os

class Path:
    _ROOT = os.getcwd()

    @staticmethod
    def _get_data_dir():
        return os.path.join(Path._ROOT, 'data')
    
    @staticmethod
    def _get_images_dir():
        return os.path.join(Path._get_data_dir(), 'images')
    
    @staticmethod
    def _get_annotations_dir():
        return os.path.join(Path._get_data_dir(), 'annotations')
    
    @staticmethod
    def get_x_train_dir():
        return os.path.join(Path._get_images_dir(), 'train')
    
    @staticmethod
    def get_x_val_dir():
        return os.path.join(Path._get_images_dir(), 'val')
    
    @staticmethod
    def get_x_test_dir():
        return os.path.join(Path._get_images_dir(), 'test')
    
    @staticmethod
    def get_y_train_dir():
        return os.path.join(Path._get_annotations_dir(), 'train')
    
    @staticmethod
    def get_y_val_dir():
        return os.path.join(Path._get_annotations_dir(), 'val')
    
    @staticmethod
    def get_y_test_dir():
        return os.path.join(Path._get_annotations_dir(), 'test')
    
    #----------------------COCO-LIKE---------------------------

    @staticmethod
    def get_y_train_file():
        return os.path.join(Path._get_annotations_dir(), 'instances_train.json')
    
    @staticmethod
    def get_y_val_file():
        return os.path.join(Path._get_annotations_dir(), 'instances_val.json')
    
    @staticmethod
    def get_y_test_file():
        return os.path.join(Path._get_annotations_dir(), 'instances_test.json')

