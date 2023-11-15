from tensorflow.keras.layers import Conv2D, Input, concatenate, BatchNormalization, Activation, Conv2DTranspose, Add
from tensorflow.keras.models import Model


def batch_activate(x, batchnorm=True):


    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def residual_block(x,
                n_filters,
                kernel_size=(3, 3), 
                batchnorm=True, 
                downsample=True,
                strides=(2, 2)):
    
    r = batch_activate(x, batchnorm)
    r = Conv2D(n_filters, kernel_size, padding='same', strides=strides)(r)
    r = batch_activate(r, batchnorm)
    r = Conv2D(n_filters, kernel_size, padding='same', strides=(1, 1))(r)

    if downsample:
        s = Conv2D(n_filters, kernel_size, padding='same', strides=strides)(x)
    else:
        s = x
    s = batch_activate(s, batchnorm)

    y = Add()([s, r])

    return y 

def first_residual_block(x,
                        n_filters,
                        kernel_size=(3, 3), 
                        batchnorm=True, 
                        downsample=True):
    
    r = Conv2D(n_filters, kernel_size, padding='same', strides=(1, 1))(x)
    r = batch_activate(r, batchnorm)
    r = Conv2D(n_filters, kernel_size, padding='same', strides=(1, 1))(r)

    if downsample:
        s = Conv2D(n_filters, kernel_size, padding='same', strides=(1, 1))(x)
    else:
        s = x
    s = batch_activate(s, batchnorm)

    y = Add()([s, r])

    return y 

def residual_unet(input_shape, 
                    n_labels,
                    n_filters=16,
                    batchnorm=True,
                    dropout=False,
                    dropout_rate=0.1):

    downsample=True

    # Input
    inputs = Input(shape=input_shape)
 
    # Contracting Path      
    r1 = first_residual_block(inputs, n_filters*1, (3, 3), batchnorm, downsample)
    
    r2 = residual_block(r1, n_filters*2, (3, 3), batchnorm, downsample)

    r3 = residual_block(r2, n_filters*4, (3, 3), batchnorm, downsample)
    
    r4 = residual_block(r3, n_filters*8, (3, 3), batchnorm, downsample)
    
    r5 = residual_block(r4, n_filters*16, (3, 3), batchnorm, downsample)

    # Bridge
    b = batch_activate(r5, batchnorm)
    c = Conv2D(n_filters*16, (3, 3), padding='same', strides=(1, 1))(b)
    b = batch_activate(c, batchnorm)
    c = Conv2D(n_filters*16, (3, 3), padding='same', strides=(1, 1))(b)
    
    # Expansive Path
    u6 = Conv2DTranspose(n_filters*16, (3, 3), strides=(2, 2), padding="same")(c)
    cc6 = concatenate([r4, u6])
    r6 = residual_block(cc6, n_filters*16, (3, 3), batchnorm, downsample, strides=(1, 1))
   
    u7 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding="same")(r6)
    cc7 = concatenate([r3, u7])
    r7 = residual_block(cc7, n_filters*8, (3, 3), batchnorm, downsample, strides=(1, 1))
    
    u8 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding="same")(r7)
    cc8 = concatenate([r2, u8])
    r8 = residual_block(cc8, n_filters*4, (3, 3), batchnorm, downsample, strides=(1, 1))
    
    u9 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding="same")(r8)
    cc9 = concatenate([r1, u9])
    r9 = residual_block(cc9, n_filters*2, (3, 3), batchnorm, downsample, strides=(1, 1))
    
    # Output
    if n_labels > 1:
        outputs = Conv2D(n_labels, (1, 1), padding='same', activation='softmax')(r9)
    elif n_labels == 1:
        outputs = Conv2D(n_labels, (1, 1), padding='same', activation='sigmoid')(r9)
    
    model = Model(inputs=inputs, outputs=outputs, name='residual_unet')
  
    return model