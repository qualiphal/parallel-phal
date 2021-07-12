import os
import json
import shutil
import numpy as np
from collections import defaultdict
from skmultilearn.model_selection import iterative_train_test_split

# Filepaths
original_imgpath   = "../data/processed_images/"
original_maskpath  = "../data/processed_masks/"
annotations_path   = "../data/annotations/instances_train.json"
labels_path        = "../data/labels.json"

new_imgpath_train  = "../data/processed_images/train/"
new_imgpath_val    = "../data/processed_images/val/"
new_imgpath_test   = "../data/processed_images/test/"
new_maskpath_train = "../data/processed_masks/train/"
new_maskpath_val   = "../data/processed_masks/val/"
new_maskpath_test  = "../data/processed_masks/test/"

# Split ratios
TRAIN_SPLIT = 0.75
VALID_SPLIT = 0.125
TEST_SPLIT  = 0.125

assert (TRAIN_SPLIT+VALID_SPLIT+TEST_SPLIT==1.0), "Splits dont add up to 100%"

if __name__=="__main__":
    img_filenames = os.listdir(original_imgpath)

    annotations = json.load(open(annotations_path, 'r'))
    id2imagename = {im['id']:'img_'+os.path.splitext(os.path.basename(im['file_name']))[0]+'.npy' for im in annotations['images']}

    label2id = json.load(open(labels_path, 'r'))

    X = []
    y = []
    d = defaultdict(set)

    for ann in annotations['annotations']:
        if id2imagename[ann['image_id']] not in img_filenames:
            continue
        d[id2imagename[ann['image_id']]].add(ann['category_id'])

    categories = sorted(list(label2id.values()))
    for name, values in d.items():
        one_hot = [0]*len(categories)
        for i, c in enumerate(categories):
            if c in values:
                one_hot[i] = 1
        X.append([name])
        y.append(one_hot)

    X = np.array(X)
    y = np.array(y)
    X_train, y_train, X_, y_ = iterative_train_test_split(X, y, test_size=1-TRAIN_SPLIT)
    X_val, y_val, X_test, y_test = iterative_train_test_split(X_, y_, test_size=TEST_SPLIT/(VALID_SPLIT+TEST_SPLIT))

    print("Train data shape:", X_train.shape, y_train.shape)
    print("Validation data shape:", X_val.shape, y_val.shape)
    print("Test data shape:", X_test.shape, y_test.shape)

    print("\nClass distribution in training dataset:", np.sum(y_train, axis=0))
    print("Class distribution in validation dataset:", np.sum(y_val, axis=0))
    print("Class distribution in testing dataset:", np.sum(y_test, axis=0))

    # Save the splitted data
    for images, dest_imgpath, dest_maskpath in (
        (list(X_train[:, 0]), new_imgpath_train, new_maskpath_train),
        (list(X_val[:, 0]), new_imgpath_val, new_maskpath_val),
        (list(X_test[:, 0]), new_imgpath_test, new_maskpath_test),
    ):
        for image_name in images:
            if not os.path.exists(dest_imgpath):
                os.makedirs(dest_imgpath)
            if not os.path.exists(dest_maskpath):
                os.makedirs(dest_maskpath)


            shutil.copy(
                src=os.path.join(original_imgpath, image_name),
                dst=os.path.join(dest_imgpath, image_name)
            )

            # Replace 'img_' with 'mask_' in beginning of name
            mask_name = 'mask_' + image_name[4:]

            shutil.copy(
                src=os.path.join(original_maskpath, mask_name),
                dst=os.path.join(dest_maskpath, mask_name)
            )
