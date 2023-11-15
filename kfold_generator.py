#install the libraries

import os
import medpy 
import numpy as np
import tensorflow as tf
from medpy.metric.binary import (dc, jc, asd, assd, precision, 
                                 recall, sensitivity, specificity) 
from tqdm.notebook import tqdm
import skimage.io as io

from Utils.Visualisation import display
from Utils.Evaluation import multiclass_metrics
from Utils.DatasetManagement import (load_images, load_masks_argmax, 
                                     save_images, save_kfold,
                                     load_kfold, load_masks,
                                     load_dicom)
from Utils.Kfold import kfold_split
from Utils.Augmentation import get_augmented
from Utils.CSVManagement import combine_results


print('imports done')


# MrOS
masks_dir = '/scratch/nk53/mi3051/mros_project/hip_300/masks/'
dcm_dir = '/scratch/nk53/mi3051/mros_project/hip_300/dicoms/'

file_names, images_all, normalised_images = load_dicom(dcm_dir)

masks_all = load_masks(masks_dir, 8, (512,512), ['cort_bone/', 'trab_bone/', 'hbm/', 'mat/', 'muscle/', 'imat/', 'sub_fat/'])

(x_train_kf, x_val_kf, x_test_kf,  
 y_train_kf, y_val_kf, y_test_kf) = kfold_split(normalised_images, 
                                                masks_all, 
                                                train_test_split_ratio=0.11, 
                                                split_shuffle=True, 
                                                k_val=10, 
                                                k_shuffle=True, 
                                                K_random_state=1)

print('kfold done')

save_kfold(x_test_kf,"/scratch/nk53/mi3051/mros_project/kfold_300_bn/test/images/")
save_kfold(y_test_kf,"/scratch/nk53/mi3051/mros_project/kfold_300_bn/test/masks/")
save_kfold(x_train_kf,"/scratch/nk53/mi3051/mros_project/kfold_300_bn/train/images/")
save_kfold(y_train_kf,"/scratch/nk53/mi3051/mros_project/kfold_300_bn/train/masks/")
save_kfold(x_val_kf,"/scratch/nk53/mi3051/mros_project/kfold_300_bn/val/images/")
save_kfold(y_val_kf,"/scratch/nk53/mi3051/mros_project/kfold_300_bn/val/masks/")

print('save done')

