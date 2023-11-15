from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Dropout, concatenate, BatchNormalization, Activation, Conv2DTranspose
from tensorflow.keras.models import Model


def dense_con_layer(x,
                    n_filters,
                    batchnorm=True,
                    dropout=True,
                    dropout_rate=0.1):

    if batchnorm:
        x = BatchNormalization()(x)
    x = Conv2D(n_filters, (1, 1), padding='same')(x)
    x = Activation('relu')(x)

    if batchnorm:
        x = BatchNormalization()(x)
    x = Conv2D(n_filters, (3, 3), padding='same')(x)
    x = Activation('relu')(x)

    if dropout:
        x = Dropout(dropout_rate)(x)

    return x


def dense_block(x, 
                n_filters,
                batchnorm=True, 
                dropout=True,
                dropout_rate=0.1):


    c1_1 = dense_con_layer(x, n_filters, batchnorm, dropout, dropout_rate)

    cc2_1 = concatenate([x, c1_1])
    c2_2 = dense_con_layer(cc2_1, n_filters, batchnorm, dropout, dropout_rate)

    cc3_1 = concatenate([x, c1_1, c2_2])
    c3_2 = dense_con_layer(cc3_1, n_filters, batchnorm, dropout, dropout_rate)

    cc4_1 = concatenate([x, c1_1, c2_2, c3_2])
    c4_2 = dense_con_layer(cc4_1, n_filters, batchnorm, dropout, dropout_rate)

    y = concatenate([x, c1_1, c2_2, c3_2, c4_2])

    return y


def transition_block(x,
                    n_filters, 
                    batchnorm=True):

    if batchnorm:
        x = BatchNormalization()(x)
    x = Conv2D(n_filters, (1, 1), padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    return x


def dense_unet(input_shape, 
                n_labels, 
                n_filters=16, 
                batchnorm=True,
                dropout=True,
                dropout_rate=0.1):


    # Input
    inputs = Input(shape=input_shape)

    #conv1 = Conv2D(n_filters*1, (3, 3), activation=None, padding="same")(inputs)

    # Contracting Path
    d1 = dense_block(inputs, n_filters*1, batchnorm, dropout, dropout_rate)
    t1 = transition_block(d1, n_filters*1, batchnorm)

    d2 = dense_block(t1, n_filters*2, batchnorm, dropout, dropout_rate)
    t2 = transition_block(d2, n_filters*2, batchnorm)

    d3 = dense_block(t2, n_filters*4, batchnorm, dropout, dropout_rate)
    t3 = transition_block(d3, n_filters*4, batchnorm)

    d4 = dense_block(t3, n_filters*8, batchnorm, dropout, dropout_rate)
    t4 = transition_block(d4, n_filters*8, batchnorm)

    # Bridge
    d5 = dense_block(t4, n_filters*16, batchnorm, dropout, dropout_rate)

    # Expansive Path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding="same")(d5)
    cc6 = concatenate([u6, d4])
    d6 = dense_block(cc6, n_filters*8, batchnorm, dropout, dropout_rate)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding="same")(d6)
    cc7 = concatenate([u7, d3])
    d7 = dense_block(cc7, n_filters*4, batchnorm, dropout, dropout_rate)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding="same")(d7)
    cc8 = concatenate([u8, d2])
    d8 = dense_block(cc8, n_filters*2, batchnorm, dropout, dropout_rate)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding="same")(d8)
    cc9 = concatenate([u9, d1])
    d9 = dense_block(cc9, n_filters*1, batchnorm, dropout, dropout_rate)

    d10 = dense_block(d9, n_filters*1, dropout_rate, batchnorm, dropout)

    # Output
    if n_labels > 1:
        outputs = Conv2D(n_labels, (1, 1), padding='same', activation='softmax')(d10)
    elif n_labels == 1:
        outputs = Conv2D(n_labels, (1, 1), padding='same', activation='sigmoid')(d10)
    
    model = Model(inputs=inputs, outputs=outputs, name='dense_unet')

    return model