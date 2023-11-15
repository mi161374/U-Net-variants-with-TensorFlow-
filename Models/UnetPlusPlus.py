from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Dropout, concatenate, BatchNormalization, Activation, Conv2DTranspose
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


def unet_pp(input_shape, 
            n_labels, 
            n_filters=16,
            batchnorm=True,
            dropout=False,
            dropout_rate=0.1,
            deep_supervision=False):

    # Input
    inputs = Input(shape=input_shape)

    # Contracting Path
    c1_1 = conv2d_block(inputs, n_filters*1, (3, 3), batchnorm, dropout, dropout_rate)
    p1_1 = MaxPooling2D((2, 2))(c1_1)

    c2_1 = conv2d_block(p1_1, n_filters*2, (3, 3), batchnorm, dropout, dropout_rate)
    p2_1 = MaxPooling2D((2, 2))(c2_1)

    c3_1 = conv2d_block(p2_1, n_filters*4, (3, 3), batchnorm, dropout, dropout_rate)
    p3_1 = MaxPooling2D((2, 2))(c3_1)

    c4_1 = conv2d_block(p3_1, n_filters*8, (3, 3), batchnorm, dropout, dropout_rate)
    p4_1 = MaxPooling2D((2, 2))(c4_1)
    
    # Bridge
    c5_1 = conv2d_block(p4_1, n_filters*16, (3, 3), batchnorm, dropout, dropout_rate)
    
    # Expansive Path
    u1_2 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same')(c2_1)
    cc1_2 = concatenate([u1_2, c1_1])
    c1_2 = conv2d_block(cc1_2, n_filters*1, (3, 3), batchnorm, dropout, dropout_rate)
    
    u2_2 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same')(c3_1)
    cc2_2 = concatenate([u2_2, c2_1])
    c2_2 = conv2d_block(cc2_2, n_filters*2, (3, 3), batchnorm, dropout, dropout_rate)
    
    u3_2 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same')(c4_1)
    cc3_2 = concatenate([u3_2, c3_1])
    c3_2 = conv2d_block(cc3_2, n_filters*4, (3, 3), batchnorm, dropout, dropout_rate)
    
    u4_2 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same')(c5_1)
    cc4_2 = concatenate([u4_2, c4_1])
    c4_2 = conv2d_block(cc4_2, n_filters*8, (3, 3), batchnorm, dropout, dropout_rate)
    
    u1_3 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same')(c2_2)
    cc1_3 = concatenate([u1_3, c1_1, c1_2])
    c1_3 = conv2d_block(cc1_3, n_filters*1, (3, 3), batchnorm, dropout, dropout_rate)
    
    u2_3 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same')(c3_2)
    cc2_3 = concatenate([u2_3, c2_1, c2_2])
    c2_3 = conv2d_block(cc2_3, n_filters*2, (3, 3), batchnorm, dropout, dropout_rate)
    
    u3_3 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same')(c4_2)
    cc3_3 = concatenate([u3_3, c3_1, c3_2])
    c3_3 = conv2d_block(cc3_3, n_filters*4, (3, 3), batchnorm, dropout, dropout_rate)

    u1_4 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same')(c2_3)
    cc1_4 = concatenate([u1_4, c1_1, c1_2, c1_3])
    c1_4 = conv2d_block(cc1_4, n_filters*1, (3, 3), batchnorm, dropout, dropout_rate)
    
    u2_4 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same')(c3_3)
    cc2_4 = concatenate([u2_4, c2_1, c2_2, c2_3])
    c2_4 = conv2d_block(cc2_4, n_filters*2, (3, 3), batchnorm, dropout, dropout_rate)

    u1_5 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same')(c2_4)
    cc1_5 = concatenate([u1_5, c1_1, c1_2, c1_3, c1_4])
    c1_5 = conv2d_block(cc1_5, n_filters*1, (3, 3), batchnorm, dropout, dropout_rate)

    # Output
    if n_labels > 1:
        activation='softmax'
    elif n_labels == 1:
        activation='sigmoid'

    nested_output1 = Conv2D(n_labels, (1, 1), padding='same', activation=activation)(c1_2)
    nested_output2 = Conv2D(n_labels, (1, 1), padding='same', activation=activation)(c1_3)
    nested_output3 = Conv2D(n_labels, (1, 1), padding='same', activation=activation)(c1_4)
    nested_output4 = Conv2D(n_labels, (1, 1), padding='same', activation=activation)(c1_5)

    if deep_supervision:
        model = Model(inputs=inputs, outputs=[nested_output1, nested_output2, 
                                            nested_output3, nested_output4])
    else:
        model = Model(inputs=inputs, outputs=nested_output4, name='unet_pp')

    return model