
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
import random
import tensorflow as tf

import os
import medpy 
import numpy as np
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

from Training.Train_gen import train_model
from Training.Test_gen import test_model
from Training.LossFunctions import dice_coef_loss


input_dir = '/scratch/nk53/mi3051/mros_project/data_png/images_normalizes/'
target_dir = '/scratch/nk53/mi3051/mros_project/data_png/masks_argmax/'
img_size = (512, 512)
num_classes = 8
batch_size = 8

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".png")
    ]
)
target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

print("Number of samples:", len(input_img_paths))


class OxfordPets(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            # y[j] -= 1
        return x, y



# Split our img paths into a training and a validation set
val_samples = 20
test_samples = 20
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_test_input_img_paths = input_img_paths[:-val_samples]
train_test_target_img_paths = target_img_paths[:-val_samples]

val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

test_input_img_paths = train_test_input_img_paths[-test_samples:]
test_target_img_paths = train_test_target_img_paths[-test_samples:]

train_input_img_paths = input_img_paths[:-test_samples]
train_target_img_paths = target_img_paths[:-test_samples]


# Instantiate data Sequences for each split
train_gen = OxfordPets(batch_size, img_size, train_input_img_paths, train_target_img_paths)
val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)
test_gen = OxfordPets(batch_size, img_size, test_input_img_paths, test_target_img_paths)

save_training_dir='/scratch/nk53/mi3051/mros_project/Training_results'
save_pred_dir='/scratch/nk53/mi3051/mros_project/Testing_results'
save_results_dir='/scratch/nk53/mi3051/mros_project/Testing_results'

kfold_num = 0

tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
strategy = tf.distribute.MirroredStrategy(devices= ['GPU:0', 'GPU:1'])

with strategy.scope():
  model = train_model(train_gen,
                      val_gen,
                      model_object=dense_unet,
                      input_shape=(512, 512, 3),
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
                      batch_size=8,
                      verbose=1)

#test_model(test_gen,
#            model,
#            save_pred=False,
#            save_pred_dir=save_pred_dir,
#            file_names=[],
#            n_labels=8,
#            metrics=[dc, jc, asd, assd, 
#                    precision, recall, sensitivity, specificity],
#            save_results=True,
#            save_results_dir=save_results_dir,
#            use_thresholding = False,
#            thresholds = [],
#            kfold_num=kfold_num)