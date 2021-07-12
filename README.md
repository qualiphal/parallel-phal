# parallel-phal

During cleaning and preprocessing, we removed the images with labels 'image_quality' and 'condition' as we only wanted to train on high quality data. Also, we would not be using 'artifact' label in training but we will keep the images, and 'pedicel' label would go into segmentation model but not in post processing.

Processed images and masks are stored as numpy arrays in npy format.
Processed masks have 6 channels (equal to number of classes defined in labels.json) and ordered in ascending order of values (category_ids)
