#!/usr/bin/env python3

# The full CNN code!
####################
from numpy.random import seed
seed(1)
from tensorflow.random import set_seed
set_seed(1)
import time
import numpy as np
from numpy import savetxt
import os
import tensorflow as tf
# import tensorflow_probability as tfp
from tensorflow.keras import backend as K  
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, AveragePooling3D, GlobalAveragePooling3D, Dense, Flatten
from tensorflow.keras.layers import Dropout, SpatialDropout3D, Activation, BatchNormalization
from tensorflow.keras.layers import Input, concatenate, add
from tensorflow.python.keras.utils import losses_utils
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
# from tensorflow.keras.layers.experimental import preprocessing
from contextlib import redirect_stdout
# from tensorflow.keras.utils import plot_model
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
# Visualization for results
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import zoom
  
t = time.time()


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[4:8], 'GPU')

#######################
LT =  0 #mD
UT = 10000 #mD

reduce_data = "false"   #true/false
trial = '_trail_1'
use_bias = "False" #True/False
data_type = 'COM'  
data_train = np.load(" ", allow_pickle=True) # load training data
data_val = np.load(" ", allow_pickle=True)   # load validation data 
data_test = np.load(" ", allow_pickle=True)  # load testing data

target = "3dir"

# Load Train set
print("Load Train data...")
print(data_train.files)
x_train = data_train['samples']
y_train = data_train['k']
casenames_train = data_train['casenames']
direction_train = data_train['direction']

info_train = np.zeros((len(y_train), 9), dtype="float64")
info_train[:, 0] = data_train['casenames']
info_train[:, 1] = data_train['porosity']
info_train[:, 2] = data_train['eff_porosity']
info_train[:, 3] = data_train['rock_type']
info_train[:, 4] = data_train['k_min']
info_train[:, 5] = data_train['k_int']
info_train[:, 6] = data_train['k_max']
info_train[:, 7] = data_train['AR']
info_train[:, 8] = data_train['DOA']
del data_train

# Load Val set
print("Load Val data...")
print(data_val.files)
x_val = data_val['samples']
y_val = data_val['k']
casenames_val = data_val['casenames']
direction_val = data_val['direction']

info_val = np.zeros((len(y_val), 9), dtype="float64")
info_val[:, 0] = data_val['casenames']
info_val[:, 1] = data_val['porosity']
info_val[:, 2] = data_val['eff_porosity']
info_val[:, 3] = data_val['rock_type']
info_val[:, 4] = data_val['k_min']
info_val[:, 5] = data_val['k_int']
info_val[:, 6] = data_val['k_max']
info_val[:, 7] = data_val['AR']
info_val[:, 8] = data_val['DOA']
del data_val

# Load Test set
print("Load Test data...")
print(data_test.files)
x_test = data_test['samples']
y_test = data_test['k']
casenames_test = data_test['casenames']
direction_test = data_test['direction']

info_test = np.zeros((len(y_test), 9), dtype="float64")
info_test[:, 0] = data_test['casenames']
info_test[:, 1] = data_test['porosity']
info_test[:, 2] = data_test['eff_porosity']
info_test[:, 3] = data_test['rock_type']
info_test[:, 4] = data_test['k_min']
info_test[:, 5] = data_test['k_int']
info_test[:, 6] = data_test['k_max']
info_test[:, 7] = data_test['AR']
info_test[:, 8] = data_test['DOA']
del data_test


print("train data length = "+str(len(y_train))+" samples")
print("val data length = "+str(len(y_val))+" samples")
print("test data length = "+str(len(y_test))+" samples")


# Reshape the images from (28, 28) to (28, 28, 1)
x_train = np.expand_dims(x_train, axis=4)
x_val = np.expand_dims(x_val, axis=4)
x_test = np.expand_dims(x_test, axis=4)


# CNN Architecture
batch_size = 32 #can be incresed by 7 (# of GPUs), but it will converge slower
epochs = 50
A = 'relu'
af = 'relu'
L = 'MAPE'

loss = tf.keras.losses.MeanAbsolutePercentageError(
     reduction=losses_utils.ReductionV2.AUTO, name='mean_absolute_percentage_error')

O = "adam"
opt = "adam" #defult LR=0.001

# metrics: #'mean_absolute_error', 'mean_absolute_percentage_error'
metrics = 'mean_absolute_percentage_error'
m = 'MAPE'

num_filters_B1 = 16
num_filters_B2 = 16
# padding = 'same' # 'valid'


save_name = 'COM_'+str(len(y_train))+'_'+str(validation_split)+'_'+O+'_'+L+'_'+str(epochs)+'_'+target+'_'+A+trial
os.mkdir(save_name)
# callbacks
csv_logger = CSVLogger(save_name+'/training.log')

checkpoint_filepath = save_name+'/checkpoint'
model_checkpoint_callback = ModelCheckpoint(
                            filepath=checkpoint_filepath,
                            save_weights_only=True,
                            monitor='val_loss',
                            mode='min',
                            save_best_only=True)

# Use with SGD, Adam already has an adaptive algorithm and its LR is the max LR
def scheduler(epoch):
  if epoch < 10:
    return 0.001
  else:
    return 0.001 * tf.math.exp(0.1 * (10 - epoch))
LRS_callback = LearningRateScheduler(scheduler)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.2, patience=5, min_lr=0)

callbacks = [csv_logger, model_checkpoint_callback]

# 2 branches
# function for creating an inception block
def inception_module(layer_in, f1, f2, strides):
	  # 7x7 conv
    conv7 = layer_in
    conv7 = Conv3D(f1, 7, strides=strides, padding='same', use_bias="False")(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv3D(f1, 7, strides=strides, padding='same', use_bias="False")(conv7)
    conv7 = Activation('relu')(conv7)
	  # 15x15 conv
    conv15 = layer_in
    conv15 = Conv3D(f2, 15, strides=strides, padding='same', use_bias="False")(conv15)
    conv15 = Activation('relu')(conv15)
    conv15 = Conv3D(f2, 15, strides=strides, padding='same', use_bias="False")(conv15)
    conv15 = Activation('relu')(conv15)
    layer_out = concatenate([conv7, conv15], axis=-1) # without skip connnection
    return layer_out

input_shape=(100, 100, 100, 1)
# Use strategy for Multiple GPUs
tf.debugging.set_log_device_placement(True)
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():

  # define model input
  input_image = Input(shape=(100, 100, 100, 1))
  # add inception module
  layer = inception_module(input_image, 16, 16, strides=2)
  layer = BatchNormalization()(layer)
  # layer = SpatialDropout3D(0.1)(layer)
  layer = Conv3D(16, 2, strides=2, padding='same', use_bias="False")(layer) #best result
  # add Conv3D block
  layer = Conv3D(32, 5, strides=1, padding='same', use_bias="False")(layer)
  layer = Activation('relu')(layer)
  # layer = BatchNormalization()(layer)
  layer = Conv3D(32, 5, strides=1, padding='same', use_bias="False")(layer)
  layer = Activation('relu')(layer)
  layer = BatchNormalization()(layer)
  layer = SpatialDropout3D(0.1)(layer)
  layer = Conv3D(32, 2, strides=2, padding='same', use_bias="False")(layer) #best result
  
  # FCL part
  layer = Flatten()(layer)
  layer = Dense(128, use_bias=use_bias, activation='relu')(layer)
  layer = Dropout(0.1)(layer)
  layer = Dense(64, use_bias=use_bias, activation='relu')(layer)
  layer = Dense(1)(layer)

  
  
  # create model
  model = Model(inputs=input_image, outputs=layer)

  model.compile(optimizer=opt, loss=loss, metrics=[metrics])


print(model.summary())
history = model.fit(
          x_train,
          y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_val,y_val),
          shuffle=True,
          callbacks=[callbacks])


# save model summary to a text file
with open(save_name+'/model_summary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

# Save the model to disk.
tf.saved_model.save(model, save_name) ##+'/model'
np.save(save_name+'/history.npy', history.history)
# The model weights (that are considered the best) are loaded into the model.
# model.load_weights(checkpoint_filepath)



# Draw history grphs for accuracy
plt.plot(history.history[metrics])
plt.plot(history.history['val_'+metrics])
plt.title('model ' + m)
# plt.yscale("logit")
plt.ylabel(m)
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.savefig(save_name+'/metric_graph.png', dpi=300)
# plt.show()
plt.clf()
# Draw history grphs for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss ' + L)
# plt.yscale("logit")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(save_name+'/loss_gragh.png', dpi=300)
# plt.show()
plt.clf()


predict_test = model.predict(x_test)
predict_val = model.predict(x_val)
predict_train = model.predict(x_train)


y_test_mD = y_test
y_train_mD = y_train
y_val_mD = y_val
predict_test_mD = predict_test
predict_train_mD = predict_train
predict_val_mD = predict_val

    
y_test_PE = np.zeros((len(predict_test)), dtype="float64")
y_test_mD_PE = np.zeros((len(predict_test)), dtype="float64")
for i in range(0, len(predict_test)):
    y_test_PE[i] = np.abs(predict_test[i]-y_test[i])/y_test[i]
    y_test_mD_PE[i] = np.abs(predict_test_mD[i]-y_test_mD[i])/y_test_mD[i]
MAPE = np.mean(y_test_PE)
MAPE_mD = np.mean(y_test_mD_PE)


y_test_mD_error = predict_test_mD[:, 0] - y_test_mD
Y_summary = np.zeros((len(y_test), 14), dtype="float64")
Y_summary[:, 0] = casenames_test
Y_summary[:, 1] = info_test[:, 0]
Y_summary[:, 2] = info_test[:, 1]
Y_summary[:, 3] = info_test[:, 2]
Y_summary[:, 4] = info_test[:, 3]
Y_summary[:, 5] = info_test[:, 4]
Y_summary[:, 6] = info_test[:, 5]
Y_summary[:, 7] = info_test[:, 6]
Y_summary[:, 8] = info_test[:, 7]
Y_summary[:, 9] = info_test[:, 8]
Y_summary[:, 10] = y_test_mD
Y_summary[:, 11] = predict_test_mD[:, 0]
Y_summary[:, 12] = y_test_mD_error
Y_summary[:, 13] = y_test_mD_PE

y_train_PE = np.zeros((len(predict_train)), dtype="float64")
y_train_mD_PE = np.zeros((len(predict_train)), dtype="float64")
for i in range(0, len(predict_train)):
    y_train_PE[i] = np.abs(predict_train[i]-y_train[i])/y_train[i]
    y_train_mD_PE[i] = np.abs(predict_train_mD[i]-y_train_mD[i])/y_train_mD[i]
MAPE_train = np.mean(y_train_PE)
MAPE_mD_train = np.mean(y_train_mD_PE)


y_train_mD_error = predict_train_mD[:, 0] - y_train_mD
Y_summary_train = np.zeros((len(y_train), 14), dtype="float64")
Y_summary_train[:, 0] = casenames_train
Y_summary_train[:, 1] = info_train[:, 0]
Y_summary_train[:, 2] = info_train[:, 1]
Y_summary_train[:, 3] = info_train[:, 2]
Y_summary_train[:, 4] = info_train[:, 3]
Y_summary_train[:, 5] = info_train[:, 4]
Y_summary_train[:, 6] = info_train[:, 5]
Y_summary_train[:, 7] = info_train[:, 6]
Y_summary_train[:, 8] = info_train[:, 7]
Y_summary_train[:, 9] = info_train[:, 8]
Y_summary_train[:, 10] = y_train_mD
Y_summary_train[:, 11] = predict_train_mD[:, 0]
Y_summary_train[:, 12] = y_train_mD_error
Y_summary_train[:, 13] = y_train_mD_PE


print("Total train samples: "+str(len(y_train)))
print("Total val samples: "+str(len(y_val)))
print("Total test samples: "+str(len(y_test)))
print("MAPE :" + str(np.round(MAPE_mD*100, 2))+" %")

time = np.round((time.time() - t)/60,1)
print("Training "+str(epochs)+" epochs completed in: "+str(time)+ " mins")

Y_summary_red = np.delete(Y_summary, indices_remove, axis=0)


# save results
savetxt(save_name+'/results_summary.csv', Y_summary, delimiter=',')
savetxt(save_name+'/results_summary_dir.csv', direction_test, delimiter=',', fmt="%s")
savetxt(save_name+'/results_summary_train.csv', Y_summary_train, delimiter=',')
savetxt(save_name+'/results_summary_dir_train.csv', direction_train, delimiter=',', fmt="%s")
