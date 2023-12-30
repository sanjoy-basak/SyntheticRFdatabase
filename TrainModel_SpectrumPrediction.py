#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 17:10:39 2023

@author: sbasak
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf

tf.config.run_functions_eagerly(False)

try:
	physical_devices = tf.config.list_physical_devices('GPU')
except: pass


from tensorflow import keras
import tensorflow as tf
import numpy as np

import h5py
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.layers import Conv2D, Input, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, LSTM, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping

plt_save=0
epochs = 1500

train = True
load_train = False


mods=5

filename= 'GenDataYOLO_prediction.h5'

h5f = h5py.File(filename, 'r')

X_train_t = h5f['X_train']
Y_train_t = h5f['Y_train']

X_test_t = h5f['X_test']
Y_test_t = h5f['Y_test']
Mod_train_t =  h5f['Mod_train']
Mod_test_t =  h5f['Mod_test']

include_label_t = h5f['include_label']
mods_t = h5f['mods']


X_train=np.array(X_train_t[()])
X_test=np.array(X_test_t[()])
Y_train=np.array(Y_train_t[()])
Y_test=np.array(Y_test_t[()])

Mod_train=np.array(Mod_train_t[()])
Mod_test=np.array(Mod_test_t[()])

include_label=bool(np.array(include_label_t[()]))
#update mod
mods=int(np.array(mods_t[()]))

h5f.close()


timeseqT = X_train.shape[1] # not present here


nsamplesTr=X_train.shape[0]
nsamplesTs=X_test.shape[0]

X_train=X_train.reshape(nsamplesTr,X_train.shape[1],X_train.shape[2],X_train.shape[3],1)
X_test=X_test.reshape(nsamplesTs,X_test.shape[1],X_test.shape[2],X_test.shape[3],1)


print("--"*10)
print("Training data IQ:",X_train.shape)
print("Training labels:",Y_train.shape)
print("Testing data IQ",X_test.shape)
print("Testing labels",Y_test.shape)
print("--"*10)


class dataprep:
    def __init__(self,num_frames,width,height,num_classes):
        self.num_frames=num_frames
        self.width=width
        self.height=height
        self.num_classes=num_classes

        

num_frames=timeseqT
width=X_train.shape[2]
height=X_train.shape[3]
num_classes=mods

data=dataprep(num_frames,width,height,num_classes)
    
def custom_loss(y_true, y_pred):
    lcord  = 5 
    lnoobj = 0.5
    pC0,pf0 = tf.split(y_pred, [1,1], axis = 1 )
    tC0,tf0 = tf.split(y_true, [1,1], axis = 1 )

    loss_xy = lcord * K.mean(K.sum(tC0 * K.square(tf0-pf0), axis=-1))

    loss_Cobj = K.mean(K.sum(tC0 * K.square(tC0-pC0), axis=-1))
    loss_Cnoobj = lnoobj * K.mean(K.sum((1-tC0) * K.square(tC0-pC0), axis=-1))
    

    tloss = loss_xy + loss_Cobj + loss_Cnoobj 
    return tloss
    

def perform_conv_nobat(x,f,kernel_sz,batch_norm,act,pad):
    if batch_norm:
        conv_2d_layer = Conv2D(f, kernel_sz, activation=None,padding=pad)        
        x = tf.keras.layers.TimeDistributed(conv_2d_layer)(x)
        x = BatchNormalization()(x)
        x= tf.keras.activations.relu(x)
    else:
        conv_2d_layer = Conv2D(f, kernel_sz, activation=act,padding=pad)
        x = (conv_2d_layer)(x)
    
    return x

def perform_maxpool_nobat(x,pad):
    maxpool_2d_layer =MaxPooling2D((2, 2), strides=(2, 2),padding=pad)
    x = tf.keras.layers.TimeDistributed(maxpool_2d_layer)(x)
    #x = (maxpool_2d_layer)(x)
    return x


def Model_Define(input_shape, classes):
    x_input = Input(input_shape)
    x = x_input
      
    batch_norm=True
    pad='valid'
    act='relu'
    kernel_sz=(3,3) #3,3
    
    x=perform_conv_nobat(x,16,kernel_sz,batch_norm,act,pad)
    x=perform_maxpool_nobat(x,pad)
    
    #add
    #x=perform_conv(x,16,(1,1),batch_norm,act,pad)    
    pad='same'    
    x=perform_conv_nobat(x,32,kernel_sz,batch_norm,act,pad)
    x=perform_maxpool_nobat(x,pad)
    
    
    kernel_sz=(4,4)
    x=perform_conv_nobat(x,64,kernel_sz,batch_norm,act,pad)
    x=perform_maxpool_nobat(x,pad)

    batch_norm=True
        
    x=perform_conv_nobat(x,128,kernel_sz,batch_norm,act,pad)
    x=perform_maxpool_nobat(x,pad)
    
    
    x=perform_conv_nobat(x,256,kernel_sz,batch_norm,act,pad)
    x=perform_maxpool_nobat(x,pad)
    
    #pad='valid'
    x=perform_conv_nobat(x,256,kernel_sz,batch_norm,act,pad)
    x=perform_maxpool_nobat(x,pad)
    
    x=perform_conv_nobat(x,512,kernel_sz,batch_norm,act,pad)
    x=perform_maxpool_nobat(x,pad)
    
    x=perform_conv_nobat(x,1024,kernel_sz,batch_norm,act,pad)
    
    pad='same'
    act='relu'
    x=perform_conv_nobat(x,125,(1,1),batch_norm,act,pad)
    
    x = TimeDistributed(Flatten())(x)    

    x = LSTM(128, return_sequences=False, dropout=0.00)(x)
    x = Dense(Y_test.shape[1]*Y_test.shape[2])(x)
    x = tf.keras.layers.Reshape((Y_test.shape[1],Y_test.shape[2]))(x)
    
    # Create model
    model = Model(inputs = x_input, outputs = x)

    return model


def evaluate_model_t2(model):
    
    batch_size_eval = batch_size
    #batch_size_eval = int(batch_size/2);
    total_batch = int(X_test.shape[0] / batch_size_eval)
    num_samps_crct = total_batch*batch_size_eval
    targets = Y_test[0:num_samps_crct]
    print('*****  Eval *****')
    
    output_pred = []
    for i in range(total_batch):
        offset = (i * batch_size) % (n_samples)
        batch_xs_input = X_test[offset:(offset + batch_size), :]
        output_pred_temp = model.predict(batch_xs_input,verbose=0)
        if i==0:
            output_pred = output_pred_temp
        else:
            output_pred= np.append(output_pred,output_pred_temp,axis=0)
    output_pred = np.array(output_pred)    
    output_loss = custom_loss(targets, output_pred)

    return output_loss.numpy()

checkpoint_path = "training_TF/CNN_LSTM.h5"

# Create a callback that saves the model's weights

cp_callback   = [
      EarlyStopping(monitor='val_loss', patience=50, mode='min', min_delta=1e-07),
      ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_best_only=True, mode='min',verbose=1)
]

opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)



model = Model_Define((data.num_frames,data.width, data.height, 1), mods)
print(model.summary())

@tf.function
def train_step(model, x, optimizer,true_labels):
    with tf.GradientTape() as tape:
        outputs = model(x)
        loss_op = custom_loss(true_labels, outputs)
    gradients = tape.gradient(loss_op, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    

# Train the model
if load_train:
    model.load_weights(checkpoint_path)

batch_size = 16
n_samples = X_train.shape[0]
total_batch = int(n_samples / batch_size)
from tqdm import tqdm

epoch_max = 20
epoch_patience = 0
if train:
    Global_loss = 1e6
    for epoch in range(1, epochs + 1):
        epoch_patience = epoch_patience+1
        for i in tqdm(range(total_batch)):
            offset = (i * batch_size) % (n_samples)
            batch_xs_input = X_train[offset:(offset + batch_size), :]
            labels_true = Y_train[offset:(offset + batch_size)]
            train_step(model, batch_xs_input, opt, labels_true)
        
        if epoch%2==0:
          eval_model = True
        else:
            eval_model = False
            
        if eval_model:
            total_eval_loss = evaluate_model_t2(model)
            if Global_loss>total_eval_loss:
                Global_loss = total_eval_loss
                model.save_weights(checkpoint_path)
                print("saving!! loss:",Global_loss)
                epoch_patience = 0
            else:
                if epoch_patience>=epoch_max:
                    print('breaking training not improving!')
                    break
            
else:
    model.load_weights(checkpoint_path)



