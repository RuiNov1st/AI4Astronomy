# -*- coding: utf-8 -*-
# Keras CNN model
# From "Photometric redshifts from SDSS images using a Convolutional Neural Network" 
# by J.Pasquet et al. 2018

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Input, Concatenate, AveragePooling2D, Flatten, PReLU, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant, GlorotUniform

def prelu(x):
    return PReLU()(x)

def conv2d(input, num_output_channels, kernel_size, name):
    conv = Conv2D(filters=num_output_channels, 
                  kernel_size=kernel_size, 
                  padding='same', 
                  kernel_initializer=GlorotUniform(), 
                  bias_initializer=Constant(0.1), 
                  name=name)(input)
    return prelu(conv)

def pool2d(input, kernel_size, stride, name):
    return AveragePooling2D(pool_size=kernel_size, strides=stride, padding='same', name=name)(input)

def fully_connected(input, num_outputs, name, withrelu=True):
    fc = Dense(units=num_outputs, 
               kernel_initializer=GlorotUniform(), 
               bias_initializer=Constant(0.1), 
               name=name)(input)
    if withrelu:
        fc = tf.nn.relu(fc)
    return fc

def inception(input, nbS1, nbS2, name, output_name, without_kernel_5=False):
    s1_0 = conv2d(input=input, num_output_channels=nbS1, kernel_size=1, name=name + "S1_0")
    s2_0 = conv2d(input=s1_0, num_output_channels=nbS2, kernel_size=3, name=name + "S2_0")

    s1_2 = conv2d(input=input, num_output_channels=nbS1, kernel_size=1, name=name + "S1_2")
    pool0 = pool2d(input=s1_2, kernel_size=2, stride=1, name=name + "pool0")

    s2_2 = conv2d(input=input, num_output_channels=nbS2, kernel_size=1, name=name + "S2_2")
    if not without_kernel_5:
        s1_1 = conv2d(input=input, num_output_channels=nbS1, kernel_size=1, name=name + "S1_1")
        s2_1 = conv2d(input=s1_1, num_output_channels=nbS2, kernel_size=5, name=name + "S2_1")
        
        output = Concatenate(name=output_name)([s2_2, s2_1, s2_0, pool0])
    else:
        output = Concatenate(name=output_name)([s2_2, s2_0, pool0])

    return output

def Pasquet_model(Augm=True,img_shape=(64,64,5),Nbins=180,reddening_correction=False,name='Qasquet2019'):
    """
        Creating a CNN using a Pasquet-like architecture.
        Essentially a GoogleNet.

        Arguments:
            Augm (Boolean): If True, it adds data augmentation to the model
            input.

            
            img_shape (numpy array): Training images' shape

            name (str): The desired name of the model.
        
        Returns:
            model(keras model): The desired GoogleNet keras model ready to 
            be compiled and trained. 

    """
    
    reddening = Input(shape=(1,), name="reddening")
    x = Input(shape=img_shape, name="x")

    if Augm:
        # Data Augmentation
        data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(factor = 0.25,fill_mode='constant',fill_value=0.), # factour:a float represented as fraction of 2 Pi,0.25*2pi = pi/2
        ])
        x = data_augmentation(x)

    conv0 = conv2d(input=x, num_output_channels=64, kernel_size=5, name="conv0")
    conv0p = pool2d(input=conv0, kernel_size=2, stride=2, name="conv0p")
    i0 = inception(conv0p, 48, 64, name="I0_", output_name="INCEPTION0")
    i1 = inception(i0, 64, 92, name="I1_", output_name="INCEPTION1")
    i1p = pool2d(input=i1, kernel_size=2, name="INCEPTION1p", stride=2)
    i2 = inception(i1p, 92, 128, name="I2_", output_name="INCEPTION2")
    i3 = inception(i2, 92, 128, name="I3_", output_name="INCEPTION3")
    i3p = pool2d(input=i3, kernel_size=2, name="INCEPTION3p", stride=2)
    i4 = inception(i3p, 92, 128, name="I4_", output_name="INCEPTION4", without_kernel_5=True)

    flat = Flatten()(i4)
    if reddening_correction:
        concat = Concatenate()([flat, reddening])
    else:
        concat = flat

    fc0 = fully_connected(input=concat, num_outputs=1096, name="fc0")
    fc1 = fully_connected(input=fc0, num_outputs=1096, name="fc0b")
    fc2 = fully_connected(input=fc1, num_outputs=Nbins, name="fc1", withrelu=False)

    output = Softmax()(fc2)
    
    if reddening_correction:
        model = Model(inputs=[x, reddening], outputs=output,name=name)
    else:
        model = Model(inputs=x, outputs=output,name=name)
        
    return model


def Treyer_model(Augm=True,img_shape=(64,64,5),Nbins=180,reddening_correction=False,name='Treyer2024'):
    """
    1. 加上regression模块
    """
    reddening = Input(shape=(1,), name="reddening")
    x = Input(shape=img_shape, name="x")

    if Augm:
        # Data Augmentation
        data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(factor = 0.25,fill_mode='constant',fill_value=0.), # factour:a float represented as fraction of 2 Pi,0.25*2pi = pi/2
        ])
        x = data_augmentation(x)

    conv0 = conv2d(input=x, num_output_channels=64, kernel_size=5, name="conv0")
    conv0p = pool2d(input=conv0, kernel_size=2, stride=2, name="conv0p")
    i0 = inception(conv0p, 48, 64, name="I0_", output_name="INCEPTION0")
    i1 = inception(i0, 64, 92, name="I1_", output_name="INCEPTION1")
    i1p = pool2d(input=i1, kernel_size=2, name="INCEPTION1p", stride=2)
    i2 = inception(i1p, 92, 128, name="I2_", output_name="INCEPTION2")
    i3 = inception(i2, 92, 128, name="I3_", output_name="INCEPTION3")
    i3p = pool2d(input=i3, kernel_size=2, name="INCEPTION3p", stride=2)
    i4 = inception(i3p, 92, 128, name="I4_", output_name="INCEPTION4", without_kernel_5=True)

    flat = Flatten()(i4)
    if reddening_correction:
        concat = Concatenate()([flat, reddening])
    else:
        concat = flat

    fc0 = fully_connected(input=concat, num_outputs=1096, name="fc0")
    # pdf:
    fc1 = fully_connected(input=fc0, num_outputs=1096, name="fc1")
    pdf_fc = fully_connected(input=fc1, num_outputs=Nbins, name="pdf_fc", withrelu=False)
    pdf_output = Softmax()(pdf_fc)
    # regression:
    fc2 = fully_connected(input=fc0, num_outputs=512, name="fc2")
    regression_output = fully_connected(input=fc2,num_outputs=1,name='regression_fc',withrelu=False)

    output = [pdf_output,regression_output]
    
    if reddening_correction:
        model = Model(inputs=[x, reddening], outputs=output,name=name)
    else:
        model = Model(inputs=x, outputs=output,name=name)
        
    return model
