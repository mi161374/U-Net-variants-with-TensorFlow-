import numpy as np
from tensorflow.keras.utils import to_categorical
import albumentations as A

transform = A.Compose([
    A.Affine(rotate=(-10,10), cval_mask=5, p=1),
    A.Affine(shear=(-5,5), cval_mask=5, p=1),
    A.Affine(scale=(0.95,1.05), cval_mask=5, p=1),
])

def get_augmented(images, 
                  masks, 
                  aug_num=3, 
                  argmax_in=True, 
                  argmax_out=False):


  images_aug = []
  masks_aug = []

  if not argmax_in:
    masks = np.argmax(masks, axis=3)
  for j in range(aug_num):
    for i in range(len(images)):
      transformed = transform(image = images[i,:,:], mask = masks[i,:,:])
      images_aug.append(transformed['image'])
      masks_aug.append(transformed['mask'])
    
  images_aug = np.asarray(images_aug)
  masks_aug = np.asarray(masks_aug)

  if not argmax_out:
    masks_aug = to_categorical(masks_aug, num_classes=6)
    
  return images_aug, masks_aug