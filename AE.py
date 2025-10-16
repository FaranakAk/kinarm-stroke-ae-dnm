"""
Implementation of the Deep Embedded Self-Organizing Map model
Autoencoder helper function

@author Florent Forest
@version 2.0
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Flatten, Dropout, Reshape, Conv2DTranspose, BatchNormalization, Activation, LayerNormalization
from tensorflow.keras.regularizers import l2

def mlp_autoencoder(encoder_dims, act='relu', init='glorot_uniform'):
    """
    Fully connected symmetric autoencoder model.

    # Arguments
        encoder_dims: list of number of units in each layer of encoder. encoder_dims[0] is input dim, encoder_dims[-1] is units in hidden layer (latent dim).
        The decoder is symmetric with encoder, so number of layers of the AE is 2*len(encoder_dims)-1
        act: activation of AE intermediate layers, not applied to Input, Hidden and Output layers
        init: initialization of AE layers
    # Return
        (ae_model, encoder_model, decoder_model): AE, encoder and decoder models
    """
    n_stacks = len(encoder_dims) - 1
    l2_value = 1e-5

    # Input
    x = Input(shape=(encoder_dims[0],), name='input')
    # Internal layers in encoder
    encoded = x
    for i in range(n_stacks-1):
        encoded = Dense(encoder_dims[i + 1], activation=act, kernel_initializer=init, activity_regularizer=l2(l2_value), name='encoder_%d' % i)(encoded)
    # Hidden layer (latent space)
    encoded = Dense(encoder_dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(encoded) # hidden layer, latent representation is extracted from here
    # Internal layers in decoder
    #encoded = LayerNormalization(axis=-1, center=True, scale=True)(encoded)
    decoded = encoded
    for i in range(n_stacks-1, 0, -1):
        decoded = Dense(encoder_dims[i], activation=act, kernel_initializer=init, activity_regularizer=l2(l2_value), name='decoder_%d' % i)(decoded)
    # Output
    decoded = Dense(encoder_dims[0], activation='sigmoid', kernel_initializer=init, name='decoder_0')(decoded) # sigmoid added by me

    # AE model
    autoencoder = Model(inputs=x, outputs=decoded, name='AE')

    # Encoder model
    encoder = Model(inputs=x, outputs=encoded, name='encoder')

    # Create input for decoder model
    encoded_input = Input(shape=(encoder_dims[-1],))
    # Internal layers in decoder
    decoded = encoded_input
    for i in range(n_stacks-1, -1, -1):
        decoded = autoencoder.get_layer('decoder_%d' % i)(decoded)
    # Decoder model
    decoder = Model(inputs=encoded_input, outputs=decoded, name='decoder')

    return autoencoder, encoder, decoder


def conv_autoencoder(encoder_dims, act='relu', init='glorot_uniform'):
    
    lat_dim = 50
    input_ae = Input(shape=(3, 300, 1))
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='e1')(input_ae)
    x = Dropout(0.2)(x)
    x = MaxPooling2D((1, 2), padding='same', name='mp1')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='e2')(x)
    x = Dropout(0.2)(x)
    x = MaxPooling2D((1, 2), padding='same', name='mp2')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='e3')(x)
    x = Dropout(0.2)(x)
    x = MaxPooling2D((1, 2), padding='same', name='mp3')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='e4')(x)
    x = Dropout(0.2)(x)
    x = MaxPooling2D((1, 2), padding='same', name='mp4')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='e5')(x)
    x = Dropout(0.2)(x)
    encoded = MaxPooling2D((1, 2), padding='same', name='mp5')(x)
#    x = Conv2D(32, (3, 3), activation='sigmoid', padding='same', name='e6')(x)
#    encoded = MaxPooling2D((1, 2), padding='same', name='mp6')(x)
    _,dim1,dim2,dim3 = encoded.shape
    encoded = Flatten(name='flat')(encoded)
    
    dim_pre_lat = int(np.sqrt(dim1*dim2*dim3*lat_dim))
    encoded = Dense(dim_pre_lat, activation='relu') (encoded)
    
    encoded = Dense(lat_dim, activation='linear', kernel_initializer=init, name='encoded')(encoded)
#    encoded = Dropout(0.5)(encoded)
#    x = encoded
#    x = Dense(dim_pre_lat, activation='relu', name='encoder_0') (x)
#    x = Dense(dim1*dim2*dim3, activation='relu', kernel_initializer=init, name='dens1')(x)
##    x = Dropout(0.5)(x)
#    x = Reshape((dim1, dim2, dim3), input_shape=(dim1*dim2*dim3,), name='reshap')(x)
##    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='d1')(x)
##    x = UpSampling2D((1, 2), name='us1')(x)
#    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='d2')(x)
##    x = Dropout(0.5, name='dr1')(x)
#    x = BatchNormalization() (x)
#    x = UpSampling2D((1, 2), name='us2')(x)
#    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='d3')(x)
##    x = Dropout(0.5, name='dr2')(x)
#    x = UpSampling2D((1, 2), name='us3')(x)
#    x = ZeroPadding2D(padding=(1,0), name='zp1')(x)
#    x = Conv2D(16, (3, 3), activation='relu', name='d4')(x)
##    x = Dropout(0.5, name='dr3')(x)
#    x = BatchNormalization() (x)
#    x = UpSampling2D((1, 2), name='us4')(x)
#    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='d5')(x)
##    x = Dropout(0.5, name='dr4')(x)
#    x = UpSampling2D((1, 2), name='us5')(x)
#    x = ZeroPadding2D(padding=(1,0), name='zp2')(x)
#    x = Conv2D(8, (3, 3), activation='relu', name='d6')(x)
##    x = Dropout(0.5, name='dr5')(x)
#    x = UpSampling2D((1, 2), name='us6')(x)
#    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='decoder_0')(x)
##    decoded = Dropout(0.5)(decoded)
    
#  ----------------------------------------------------------  
    x = encoded
    x = Dense(dim_pre_lat, activation='relu', name='encoder_0') (x)
    x = Dense(dim1*dim2*dim3, activation='relu', kernel_initializer=init, name='dens1')(x)
#    x = Dropout(0.5)(x)
    x = Reshape((dim1, dim2, dim3), input_shape=(dim1*dim2*dim3,), name='reshap')(x)
#    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='d1')(x)
#    x = UpSampling2D((1, 2), name='us1')(x)
    x = Conv2DTranspose(32, (3, 3), strides=(1,2), activation='linear', padding='same', name='d2')(x)
    x = BatchNormalization() (x)
    x = Activation('relu') (x)

    x = Conv2DTranspose(16, (3, 3), strides=(1, 2), activation='relu', padding='same', name='d3')(x)

    x = ZeroPadding2D(padding=(1,0), name='zp1')(x)
    x = Conv2D(16, (3, 3), activation='relu', name='d4')(x)
    x = UpSampling2D((1, 2), name='us4')(x)
    
    x = Conv2DTranspose(8, (3, 3), strides=(1,2), activation='linear', padding='same')(x)
    x = BatchNormalization() (x)
    x = Activation('relu') (x)
    
    x = ZeroPadding2D(padding=(1,0), name='zp2')(x)
    x = Conv2D(8, (3, 3), activation='relu', name='d6')(x)
    x = UpSampling2D((1, 2), name='us6')(x)
    
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='decoder_0')(x)
#    decoded = Dropout(0.5)(decoded)
    
    
    autoencoder = Model(inputs=input_ae, outputs=decoded, name='AE')
    
    encoder = Model(inputs=input_ae, outputs=encoded, name='encoder')
    
    encoded_input = Input(shape=(lat_dim,))
    decoded = encoded_input
    for i,layer in enumerate(autoencoder.layers):
        if layer.name=='encoder_0':
            ind_start = i
        elif layer.name =='decoder_0':
            ind_end = i+1
            break
    
    for i in range(ind_start,ind_end):
        decoded = autoencoder.get_layer(index=i) (decoded)
    
#    encoded_input = Input(shape=(10,))
#    decoded = encoded_input
#    decoded = autoencoder.get_layer('dens1')(decoded)
#    decoded = autoencoder.get_layer('reshap')(decoded)
##    decoded = autoencoder.get_layer('d1')(decoded)
##    decoded = autoencoder.get_layer('us1')(decoded)
#    decoded = autoencoder.get_layer('d2')(decoded)
#    decoded = autoencoder.get_layer('dr1')(decoded)
#    decoded = autoencoder.get_layer('us2')(decoded)
#    decoded = autoencoder.get_layer('d3')(decoded)
#    decoded = autoencoder.get_layer('dr2')(decoded)
#    decoded = autoencoder.get_layer('us3')(decoded)
#    decoded = autoencoder.get_layer('zp1')(decoded)
#    decoded = autoencoder.get_layer('d4')(decoded)
#    decoded = autoencoder.get_layer('dr3')(decoded)
#    decoded = autoencoder.get_layer('us4')(decoded)
#    decoded = autoencoder.get_layer('d5')(decoded)
#    decoded = autoencoder.get_layer('dr4')(decoded)
#    decoded = autoencoder.get_layer('us5')(decoded)
#    decoded = autoencoder.get_layer('zp2')(decoded)
#    decoded = autoencoder.get_layer('d6')(decoded)
#    decoded = autoencoder.get_layer('dr5')(decoded)
#    decoded = autoencoder.get_layer('us6')(decoded)
#    decoded = autoencoder.get_layer('decoder_0')(decoded)
#    
#    
    decoder = Model(inputs=encoded_input, outputs=decoded, name='decoder')
    
    return autoencoder, encoder, decoder
#    return autoencoder, encoder, encoder
    



def conv_autoencoder_256(encoder_dims, act='relu', init='glorot_uniform'):
    
    lat_dim = 50
    input_ae = Input(shape=(3, 256, 1))
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='e1')(input_ae)
    x = Dropout(0.2)(x)
    x = MaxPooling2D((1, 2), padding='same', name='mp1')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='e2')(x)
    x = Dropout(0.2)(x)
    x = MaxPooling2D((1, 2), padding='same', name='mp2')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='e3')(x)
    x = Dropout(0.2)(x)
    x = MaxPooling2D((1, 2), padding='same', name='mp3')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='e4')(x)
    x = Dropout(0.2)(x)
    x = MaxPooling2D((1, 2), padding='same', name='mp4')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='e5')(x)
    x = Dropout(0.2)(x)
    encoded = MaxPooling2D((1, 2), padding='same', name='mp5')(x)
    _,dim1,dim2,dim3 = encoded.shape
    encoded = Flatten(name='flat')(encoded)
    
    dim_pre_lat = int(np.sqrt(dim1*dim2*dim3*lat_dim))
    encoded = Dense(dim_pre_lat, activation='relu') (encoded)
    
    encoded = Dense(lat_dim, activation='linear', kernel_initializer=init, name='encoded')(encoded)

#  ----------------------------------------------------------  
    x = encoded
    x = Dense(dim_pre_lat, activation='relu', name='encoder_0') (x)
    x = Dense(dim1*dim2*dim3, activation='relu', kernel_initializer=init, name='dens1')(x)
    x = Reshape((dim1, dim2, dim3), input_shape=(dim1*dim2*dim3,), name='reshap')(x)
    x = Conv2DTranspose(32, (3, 3), strides=(1,2), activation='linear', padding='same', name='d2')(x)
    x = BatchNormalization() (x)
    x = Activation('relu') (x)

    x = Conv2DTranspose(16, (3, 3), strides=(1, 2), activation='relu', padding='same')(x)
    x = Conv2DTranspose(16, (3, 3), strides=(1, 2), activation='relu', padding='same')(x)

    
    x = Conv2DTranspose(8, (3, 3), strides=(1,2), activation='linear', padding='same')(x)
    x = BatchNormalization() (x)
    x = Activation('relu') (x)
    
    x = Conv2DTranspose(8, (3, 3), strides=(1, 2), activation='relu', padding='same')(x)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='decoder_0')(x)
#    decoded = Dropout(0.5)(decoded)
    
    
    autoencoder = Model(inputs=input_ae, outputs=decoded, name='AE')
    
    encoder = Model(inputs=input_ae, outputs=encoded, name='encoder')
    
    encoded_input = Input(shape=(lat_dim,))
    decoded = encoded_input
    for i,layer in enumerate(autoencoder.layers):
        if layer.name=='encoder_0':
            ind_start = i
        elif layer.name =='decoder_0':
            ind_end = i+1
            break
    
    for i in range(ind_start,ind_end):
        decoded = autoencoder.get_layer(index=i) (decoded)
        
    decoder = Model(inputs=encoded_input, outputs=decoded, name='decoder')
    
    return autoencoder, encoder, decoder