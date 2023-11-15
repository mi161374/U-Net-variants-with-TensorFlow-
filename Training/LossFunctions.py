import keras
import keras.backend as K
import tensorflow as tf


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_true_f_cast = tf.cast(y_true_f, tf.float32)
    intersection = K.sum(y_true_f_cast * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f_cast) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)