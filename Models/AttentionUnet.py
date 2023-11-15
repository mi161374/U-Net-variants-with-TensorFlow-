from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, concatenate, BatchNormalization, Activation, Conv2DTranspose, Add, Multiply, Dropout
from tensorflow.keras.models import Model

def batch_activate(x, batchnorm=True):


    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def conv2d_block(x,
                n_filters, 
                kernel_size=(3, 3), 
                batchnorm=True,
                dropout=False,
                dropout_rate=0.1):
    

    x = Conv2D(n_filters, kernel_size, padding='same')(x)
    x = batch_activate(x, batchnorm)

    x = Conv2D(n_filters, kernel_size, padding='same')(x)
    x = batch_activate(x, batchnorm)

    if dropout:
        x = Dropout(dropout_rate)(x)

    return x


def attention_block(x1, 
                    x2, 
                    n_filters,
                    batchnorm=True):


    g = Conv2D(n_filters, (1, 1), strides=(1, 1), padding='same')(x1)
    if batchnorm:
        g = BatchNormalization()(g)

    x = Conv2D(n_filters, (1, 1), strides=(1, 1), padding='same')(x2)
    if batchnorm:
        x = BatchNormalization()(x)

    p = Add()([g, x])
    p = Activation('relu')(p)
    p = Conv2D(n_filters, (1, 1), strides=(1, 1), padding='same')(p)
    if batchnorm:
        p = BatchNormalization()(p)
    p = Activation('sigmoid')(p)

    y = Multiply()([x2, p])

    return y
    

def attention_unet(input_shape, 
                    n_labels, 
                    n_filters=16,
                    batchnorm=True,      
                    dropout=False,
                    dropout_rate=0.1):
    
    # Inputs
    inputs = Input(shape=input_shape)


    # Contracting Path
    c1 = conv2d_block(inputs, n_filters*1, (3, 3), batchnorm, dropout, dropout_rate)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = conv2d_block(p1, n_filters*2, (3, 3), batchnorm, dropout, dropout_rate)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = conv2d_block(p2, n_filters*4, (3, 3), batchnorm, dropout, dropout_rate)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = conv2d_block(p3, n_filters*8, (3, 3), batchnorm, dropout, dropout_rate)
    p4 = MaxPooling2D((2, 2))(c4)
    
    # Bridge
    c5 = conv2d_block(p4, n_filters*16, (3, 3), batchnorm, dropout, dropout_rate)

    # Expansive Path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same')(c5)
    a6 = attention_block(c4, u6, n_filters*8, batchnorm)
    cc6 = concatenate([u6, a6])
    c6 = conv2d_block(cc6, n_filters*8, (3, 3), batchnorm, dropout, dropout_rate)
    
    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same')(c6)
    a7 = attention_block(c3, u7, n_filters*4, batchnorm)
    cc7 = concatenate([u7, a7])
    c7 = conv2d_block(cc7, n_filters*4, (3, 3), batchnorm, dropout, dropout_rate)
    
    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same')(c7)
    a8 = attention_block(c2, u8, n_filters*2, batchnorm)
    cc8 = concatenate([u8, a8])
    c8 = conv2d_block(cc8, n_filters*2, (3, 3), batchnorm, dropout, dropout_rate)
    
    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same')(c8)
    a9 = attention_block(c1, u9, n_filters*1, batchnorm)
    cc9 = concatenate([u9, a9])
    c9 = conv2d_block(cc9, n_filters*1, (3, 3), batchnorm, dropout=False)

    # Output
    if n_labels > 1:
        outputs = Conv2D(n_labels, (1, 1), padding='same', activation='softmax')(c9)
    elif n_labels == 1:
        outputs = Conv2D(n_labels, (1, 1), padding='same', activation='sigmoid')(c9)

    model = Model(inputs=inputs, outputs=outputs, name='attention_unet')

    return model