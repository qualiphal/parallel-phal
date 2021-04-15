import numpy as np
import cv2

def getNormalMask(coco, imageObj, filterClasses):
    """
    iscrowd is set to None, therefore it only works for single
    mask : (height, width)

    Parameters
    ------------------------------------
    """
    # Load categorical ids for filterclasses
    catIds = coco.getCatIds(catNms=filterClasses)
    input_image_size = (imageObj['height'],imageObj['width'])

    # Load annotations for image object
    annIds = coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    
    # Create mask for image object
    mask = np.zeros(input_image_size)
    for a in range(len(anns)):
        pixel_value = anns[a]['category_id']
        new_mask = cv2.resize(coco.annToMask(anns[a])*pixel_value, input_image_size)
        mask = np.maximum(new_mask, mask)

    return mask