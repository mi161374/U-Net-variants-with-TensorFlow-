import os
import tensorflow as tf
import numpy as np
import skimage.io as io
from tqdm.notebook import tqdm
import pydicom as dicom



def load_masks(load_path,
              n_class,
              target_size,
              class_names):

  All_GT_ims = []
  for files in sorted(os.listdir(load_path+class_names[0])):
    GT_im=np.zeros(np.concatenate((target_size,n_class),axis=None),dtype=int)
    FG=np.zeros(target_size)
    for i,cn in enumerate(class_names):
        img = io.imread(load_path+cn+files, as_gray=True)
        img = img[tf.newaxis,:,:,tf.newaxis]
        img = tf.image.resize(img,target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, antialias=True)
        img = tf.squeeze(img)
        bw = np.zeros(target_size)
        bw[img>0.5]=1
        GT_im[:,:,i] = bw
        if(np.sum(GT_im[:,:,i])>0):
          GT_im[:,:,i]=GT_im[:,:,i]/np.ptp(GT_im[:,:,i])
        FG=(FG+GT_im[:,:,i]>0).astype(int)
    GT_im[:,:,n_class-1]=1-FG
    All_GT_ims.append(GT_im)
  return np.array(All_GT_ims)

def load_images(load_path, 
                target_size):


  All_ims = []
  for files in tqdm(sorted(os.listdir(load_path))):
    img = io.imread(load_path+files, as_gray=True)
    img = img[tf.newaxis,:,:,tf.newaxis]
    img = tf.image.resize(img,target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, antialias=True)
    img = tf.squeeze(img)
    All_ims.append(img)
  names = [files.split('.',1)[0] for files in sorted(os.listdir(load_path))]
  return names, np.array(All_ims)

def load_masks_argmax(load_path, 
                      target_size):


  All_ims = []
  for files in tqdm(sorted(os.listdir(load_path))):
    img = io.imread(load_path+files)
    img = img[tf.newaxis,:,:,tf.newaxis]
    img = tf.image.resize(img,target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, antialias=True)
    img = tf.squeeze(img)
    All_ims.append(img)
  return np.array(All_ims)

def save_images(images, 
                write_path, 
                file_names):

  if not os.path.exists(write_path):
    os.makedirs(write_path) 
  for i, files in enumerate(tqdm(file_names)):
    io.imsave(os.path.join(write_path, files.split('.',1)[0]+'.tiff'), images[i,:,:])

def save_kfold(images, 
                write_path):

  if not os.path.exists(write_path):
    os.makedirs(write_path)  
  for i in range(len(images)):
    io.imsave(os.path.join(write_path, str(i) +'.tiff'), images[i])

def load_kfold(kfold_dir, k_range):
  x_train_kf = []
  x_val_kf = []
  x_test_kf = []
  y_train_kf = []
  y_val_kf = []
  y_test_kf = []
  dir = ['train/images/', 'val/images/', 'test/images/', 
         'train/masks/', 'val/masks/', 'test/masks/']

  for kfold_num in range(k_range[0], k_range[1]+1):
    x_train_kf.append(io.imread(os.path.join(kfold_dir, dir[0] + str(kfold_num) + '.tiff')))
    x_val_kf.append(io.imread(os.path.join(kfold_dir, dir[1] + str(kfold_num) + '.tiff')))
    x_test_kf.append(io.imread(os.path.join(kfold_dir, dir[2] + str(kfold_num) + '.tiff')))
    y_train_kf.append(io.imread(os.path.join(kfold_dir, dir[3] + str(kfold_num) + '.tiff')))
    y_val_kf.append(io.imread(os.path.join(kfold_dir, dir[4] + str(kfold_num) + '.tiff')))
    y_test_kf.append(io.imread(os.path.join(kfold_dir, dir[5] + str(kfold_num) + '.tiff')))

  return x_train_kf, x_val_kf, x_test_kf, y_train_kf, y_val_kf, y_test_kf

def import_dcm(path):

    slices = [dicom.read_file(path + '/' + s) for s in sorted(os.listdir(path))]
    names = [files.split('.',1)[0] for files in sorted(os.listdir(path))]

    return names, slices

def load_dicom(folder_dir):
  dcm_names, dcm_images = import_dcm(folder_dir)


  rescale_slope = dcm_images[0].RescaleSlope
  rescale_intercept = dcm_images[0].RescaleIntercept

  dcm_image_arrays = np.array([dcm_images[i].pixel_array for i in range(len(dcm_images))])

  hu_image_arrays = rescale_slope * dcm_image_arrays + rescale_intercept
  hu_image_arrays[hu_image_arrays<-1000] = -1000

  norm_images = np.array([(hu_image_arrays[i] - np.min(hu_image_arrays[i])) / (np.max(hu_image_arrays[i]) - np.min(hu_image_arrays[i])) for i in range(len(hu_image_arrays))])

  return dcm_names, hu_image_arrays, norm_images