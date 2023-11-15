#install the libraries

import os
import medpy 
import numpy as np
import tensorflow as tf
from medpy.metric.binary import (dc, jc, asd, assd, precision, 
                                 recall, sensitivity, specificity) 
from tqdm.notebook import tqdm
import skimage.io as io

from tensorflow.keras import backend as k
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, 
                                        ReduceLROnPlateau, TensorBoard, 
                                        LearningRateScheduler)
from tensorflow.keras.utils import to_categorical

from Utils.Visualisation import display
from Utils.Evaluation import multiclass_metrics
from Utils.DatasetManagement import (load_images, load_masks_argmax, 
                                     save_images, save_kfold,
                                     load_kfold, load_masks,
                                     load_dicom)
from Utils.Kfold import kfold_split
from Utils.Augmentation import get_augmented
from Utils.CSVManagement import combine_results

from Models.Unet import unet
from Models.AttentionUnet import attention_unet
from Models.DenseUnet import dense_unet
from Models.InceptionUnet import inception_unet
from Models.ResidualUnet import residual_unet
from Models.UnetPlusPlus import unet_pp
from Models.WideUnet import wide_unet

from Training.Train import train_model
from Training.Test import test_model
from Training.LossFunctions import dice_coef_loss

print('imports done')

kfold_dir='/scratch/nk53/mi3051/mros_project/kfold_300_bn/'

(x_train_kf, x_val_kf, x_test_kf, 
 y_train_kf, y_val_kf, y_test_kf) = load_kfold(kfold_dir=kfold_dir, k_range=[0,9])

print('kfold loaded')


def kfold_train_test(k=[0, 10]):

  save_training_dir='/scratch/nk53/mi3051/mros_project/Training_results_300'
  save_pred_dir='/scratch/nk53/mi3051/mros_project/Testing_results_300'
  save_results_dir='/scratch/nk53/mi3051/mros_project/Testing_results_300'

  for kfold_num in range(k[0], k[1]):
    model = train_model(x_train_kf[kfold_num], 
                        y_train_kf[kfold_num], 
                        x_val_kf[kfold_num], 
                        y_val_kf[kfold_num],
                        model_object=dense_unet,
                        input_shape=(512, 512, 1),
                        n_labels=8,
                        n_filters=16,
                        batchnorm=True,
                        dropout=False,
                        dropout_rate=0.1,
                        deep_supervision=False,
                        optimizer=Adam,
                        learning_rate=1e-4,
                        loss_fnc="categorical_crossentropy",
                        kfold_num=kfold_num,
                        save_model=True,
                        save_logs=True,
                        save_dir=save_training_dir,
                        epochs=200,
                        batch_size=2,
                        verbose=1)
    
    test_model(x_test_kf[kfold_num],
              y_test_kf[kfold_num],
              model,
              save_pred=False,
              save_pred_dir=save_pred_dir,
              file_names=[],
              n_labels=8,
              metrics=[dc, jc, asd, assd, 
                        precision, recall, sensitivity, specificity],
              save_results=True,
              save_results_dir=save_results_dir,
              use_thresholding = False,
              thresholds = [],
              kfold_num=kfold_num)
              

#strategy = tf.distribute.MirroredStrategy(devices= ['GPU:0', 'GPU:1'])

#with strategy.scope():
kfold_train_test(k=[8, 10])
#tf.debugging.set_log_device_placement(True)

#with tf.device('GPU:1'):

