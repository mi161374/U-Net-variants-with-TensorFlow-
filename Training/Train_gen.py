import os
from tensorflow.keras import backend as k
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, 
                                        ReduceLROnPlateau, TensorBoard, 
                                        LearningRateScheduler)
from Models.Unet import unet
from Models.AttentionUnet import attention_unet
from Models.DenseUnet import dense_unet
from Models.InceptionUnet import inception_unet
from Models.ResidualUnet import residual_unet
from Models.UnetPlusPlus import unet_pp

def train_model(train_gen,
                val_gen,
                model_object=unet,
                input_shape=(256, 256, 1),
                n_labels=6,
                n_filters=16,
                batchnorm=True,
                dropout=False,
                dropout_rate=0.1,
                deep_supervision=False,
                optimizer=Adam,
                learning_rate=1e-4,
                loss_fnc="categorical_crossentropy",
                kfold_num=0,
                save_model=True,
                save_logs=True,
                save_dir='/content/drive/MyDrive/',
                epochs=200,
                batch_size=16, 
                verbose=0,
                summary=False):
  

    # Compile model 
    assert isinstance(input_shape, tuple), 'input_shape must a tuple: (height, width, depth)'
    assert not isinstance(model_object, str), 'model_object must an object not a name'
    assert not isinstance(optimizer, str), 'optimizer must a function not a name'

    k.clear_session()

    if model_object==unet_pp:
        model = model_object(input_shape, 
                            n_labels, 
                            n_filters, 
                            batchnorm, 
                            dropout, 
                            dropout_rate,
                            deep_supervision)
    else:
        model = model_object(input_shape, 
                            n_labels, 
                            n_filters, 
                            batchnorm, 
                            dropout, 
                            dropout_rate)

    if summary:
        model.summary()

    model.compile(optimizer=optimizer(learning_rate),
                loss=loss_fnc, 
                metrics="accuracy")

    # Create callbacks 
    callbacks = [
    #EarlyStopping(monitor = 'val_loss', patience=20, verbose=1, min_delta=0.0005),
    #ModelCheckpoint('model-Unet.h5', verbose=1, save_best_only=True, save_weights_only=True),
    #ReduceLROnPlateau(monitor = 'val_loss', factor=0.1, patience=5, min_lr=0.00001, verbose=1),
                ]

    if save_logs:
        log_path = os.path.join(save_dir, 'Saved_logs')
        model_log_path = os.path.join(log_path, model.name + '_' + str(kfold_num+1))

        if not os.path.exists(model_log_path):
            os.makedirs(model_log_path)
        callbacks.append(TensorBoard(log_dir=model_log_path))



    # Fit the model 
    model.fit(train_gen,
            epochs=epochs,
            verbose=verbose,
            validation_data=val_gen,
            callbacks=callbacks)

    print(model.name + '_' + str(kfold_num+1) + ' trained successfully')

    if save_model:
        models_path = os.path.join(save_dir, 'Saved_models')
        models_save_path = os.path.join(models_path, model.name + '_' + str(kfold_num+1))

        if not os.path.exists(models_save_path):
            os.makedirs(models_save_path)
        model.save(models_save_path)

    return model