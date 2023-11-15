from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Dropout, concatenate, BatchNormalization, Activation, Conv2DTranspose
from tensorflow.keras.models import Model

def batch_activate(x, batchnorm=True):


    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def inception_module(x, 
                    n_filters, 
                    batchnorm=True):
    

    t1 = Conv2D(n_filters, (1, 1), padding='same')(x)
    t1 = batch_activate(t1, batchnorm)
    
    t2 = Conv2D(n_filters, (1, 1), padding='same')(x)
    t2 = batch_activate(t2, batchnorm)
    t2 = Conv2D(n_filters, (3, 3), padding='same')(t2)
    t2 = batch_activate(t2, batchnorm)
    
    t3 = Conv2D(n_filters, (1, 1), padding='same')(x)
    t3 = batch_activate(t3, batchnorm)
    #t3 = Conv2D(n_filters, (3, 3), padding='same')(t3)
    #t3 = Conv2D(n_filters, (3, 3), padding='same')(t3)
    t3 = Conv2D(n_filters, (5, 5), padding='same')(t3)
    t3 = batch_activate(t3, batchnorm)
    
    t4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    t4 = Conv2D(n_filters, (1, 1), padding='same')(t4)
    t4 = batch_activate(t4, batchnorm)
    
    y = concatenate([t1, t2, t3, t4])

    return y

def conv_dropout(x, 
                n_filters, 
                dropout=True,
                dropout_rate=0.1):


    x = Conv2D(n_filters, (1, 1), padding='same')(x)
    if dropout:
        x = Dropout(dropout_rate)(x)

    return x


def inception_unet(input_shape, 
                    n_labels, 
                    n_filters=16, 
                    batchnorm=True,
                    dropout=True,
                    dropout_rate=0.1):
    

    # Input
    inputs = Input(shape=input_shape)
    
    # Contracting Path
    i1 = inception_module(inputs, n_filters*1, batchnorm)
    cd1 = conv_dropout(i1, n_filters*1, dropout, dropout_rate)
    p1 = MaxPooling2D((2, 2))(cd1)
    
    i2 = inception_module(p1, n_filters*2, batchnorm)
    cd2 = conv_dropout(i2, n_filters*2, dropout, dropout_rate)
    p2 = MaxPooling2D((2, 2))(cd2)
    
    i3 = inception_module(p2, n_filters*4, batchnorm)
    cd3 = conv_dropout(i3, n_filters*4, dropout, dropout_rate)
    p3 = MaxPooling2D((2, 2))(cd3)
    
    i4 = inception_module(p3, n_filters*8, batchnorm)
    cd4 = conv_dropout(i4, n_filters*8, dropout, dropout_rate)
    p4 = MaxPooling2D((2, 2))(cd4)

    # Bridge
    i5 = inception_module(p4, n_filters*16, batchnorm)

    # Expansive Path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding="same")(i5)
    cc6 = concatenate([cd4, u6])
    i6 = inception_module(cc6, n_filters*8, batchnorm)
    cd6 = conv_dropout(i6, n_filters*8, dropout, dropout_rate)
    
    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding="same")(cd6)
    cc7 = concatenate([cd3, u7])
    i7 = inception_module(cc7, n_filters*4, batchnorm)
    cd7 = conv_dropout(i7, n_filters*4, dropout, dropout_rate)
    
    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding="same")(cd7)
    cc8 = concatenate([cd2, u8])
    i8 = inception_module(cc8, n_filters*2, batchnorm)
    cd8 = conv_dropout(i8, n_filters*2, dropout, dropout_rate)
    
    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding="same")(cd8)
    cc9 = concatenate([cd1, u9])
    i9 = inception_module(cc9, n_filters*1, batchnorm)
    cd9 = conv_dropout(i9, n_filters*1, dropout=False)

    # Output
    if n_labels > 1:
        outputs = Conv2D(n_labels, (1, 1), padding='same', activation='softmax')(cd9)
    elif n_labels == 1:
        outputs = Conv2D(n_labels, (1, 1), padding='same', activation='sigmoid')(cd9)
    
    model = Model(inputs=inputs, outputs=outputs, name='inception_unet')
 
    return model