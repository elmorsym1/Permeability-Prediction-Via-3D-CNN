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
from functions import inception_module, conv_module, dense_module


print(K.image_data_format()) #defult "channels_last"
# K.set_image_data_format('channels_last') #'channels_first'  
t = time.time()
# os.chdir('D:/Canada/University/PhD/Research/Programs/Python/CNN/codes')


# Choose GPU to use (8 available) - skip 0 (busy)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[4:8], 'GPU')
#######################
LT = 200  #mD
UT = 10000 #mD
validation_split = 0.2
test_split = 0.2
reduce_data = "false"   #true/false
trial = '_Test'
use_bias = "False" #True/False
# Options:
# tragets [porosity, eff. Porosity, kx, ky, kz, Resolution]
data_type = 'COM'  # Not working update "save_name" manually
data = np.load("datasets/balanced/dataset_COM_BSS_KLS_BrSS_ELS_C1_C2_len_19005_size_100_bal_200_100_9600_len_65248.npz", allow_pickle=True) #update
target = "3dir"
print(data.files)
x_train = data['samples']
y_train = data['k']
casenames = data['casenames']
direction = data['direction']
porosity = data['porosity']
eff_porosity = data['eff_porosity']
rock_type = data['rock_type']
k_min = data['k_min']
k_int = data['k_int']
k_max = data['k_max']
AR = data['AR']
DOA = data['DOA']
del data



info = np.zeros((len(y_train), 9), dtype="float64")
info[:, 0] = casenames
info[:, 1] = porosity
info[:, 2] = eff_porosity
info[:, 3] = rock_type
info[:, 4] = k_min
info[:, 5] = k_int
info[:, 6] = k_max
info[:, 7] = AR
info[:, 8] = DOA


data_len = len(y_train)



# Reduce data
if reduce_data == "true":
  indices = np.arange(0, data_len)
  indi_rem_1 = np.array(np.where(y_train < LT), dtype=int)
  indi_rem_2 = np.array(np.where(y_train > UT), dtype=int)
  indices_remove = np.append(indi_rem_1, indi_rem_2)
  del indi_rem_1, indi_rem_2
  indices_reduced = np.delete(indices, indices_remove, axis=0)
  x_train = x_train[indices_reduced, :, :, :]
  y_train = y_train[indices_reduced]
  casenames = casenames[indices_reduced]
  direction = direction[indices_reduced]
  info = info[indices_reduced, :]
  data_len = len(y_train)





print("Data length = "+str(len(y_train))+" samples")

# split dataset into train and test
# Split Train-Test
test_index = np.random.choice(len(y_train), int(test_split*len(y_train)), replace=False)
x_test = x_train[test_index, :, :, :]
y_test = y_train[test_index]
casenames_test = casenames[test_index]
direction_test = direction[test_index]
info_test = info[test_index, :]

x_train = np.delete(x_train, test_index, axis=0)
y_train = np.delete(y_train, test_index, axis=0)
casenames_train = np.delete(casenames, test_index, axis=0)
direction_train = np.delete(direction, test_index, axis=0)
info_train = np.delete(info, test_index, axis=0)

del casenames, info, direction


# Split Train-Validation
val_index = np.random.choice(len(y_train), int(validation_split*len(y_train)), replace=False)
x_val = x_train[val_index, :, :, :]
y_val = y_train[val_index]
casenames_val = casenames_train[val_index]
direction_val = direction_train[val_index]
info_val = info_train[val_index, :]

x_train = np.delete(x_train, val_index, axis=0)
y_train = np.delete(y_train, val_index, axis=0)
casenames_train = np.delete(casenames_train, val_index, axis=0)
direction_train = np.delete(direction_train, val_index, axis=0)
info_train = np.delete(info_train, val_index, axis=0)

print("train data length = "+str(len(y_train))+" samples")
print("val data length = "+str(len(y_val))+" samples")
print("test data length = "+str(len(y_test))+" samples")


# Reshape the images from (28, 28) to (28, 28, 1)
x_train = np.expand_dims(x_train, axis=4)
x_val = np.expand_dims(x_val, axis=4)
x_test = np.expand_dims(x_test, axis=4)

p1 = np.percentile(y_train, 1)
p99 = np.percentile(y_train, 99)

def above_percentile(x, p): #assuming the input is flattened: (n,)

    samples = K.cast(K.shape(x)[0], K.floatx()) #batch size
    p =  (100. - p)/100.  #100% will return 0 elements, 0% will return all elements

    #samples to get:
    samples = K.cast(tf.math.floor(p * samples), 'int32')
        #you can choose tf.math.ceil above, it depends on whether you want to
        #include or exclude one element. Suppose you you want 33% top,
        #but it's only possible to get exactly 30% or 40% top:
        #floor will get 30% top and ceil will get 40% top.
        #(exact matches included in both cases)

    #selected samples
    values, indices = tf.math.top_k(x, samples)

    return values

# Define Custome Loss function
def custom_loss_function(y_true, y_pred):
    abs = tf.abs(y_true - y_pred)
    ABS = tf.reduce_mean(abs, axis=-1)/(p99-p1)
    return ABS

# Define Custome Loss function MAPE_95
def custom_MAPE_95(y_true, y_pred):
    AE = 100 * tf.abs(y_true - y_pred) / y_true
    # p95 = np.percentile(AE, 95)
    # AE_95 = AE[tf.where(AE < p95)]
    AE_95 = -1 * above_percentile(K.flatten(-1 * AE), 5)
    MAPE_95 = tf.reduce_mean(AE_95, axis=-1)
    return MAPE_95


batch_size = 32 #can be incresed by 7 (# of GPUs), but it will converge slower
epochs = 100
A = 'relu'
af = 'relu'
# af = tf.keras.layers.LeakyReLU(alpha=0.3)
# loss: #'mean_squared_error' #'mean_absolute_error'
# loss = custom_loss_function
L = 'MAPE'

# loss = custom_MAPE_95
loss = tf.keras.losses.MeanAbsolutePercentageError(
     reduction=losses_utils.ReductionV2.AUTO, name='mean_absolute_percentage_error')


#loss = tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
# loss = tf.keras.losses.Huber(
#     delta=1.0, reduction=losses_utils.ReductionV2.AUTO, name='huber_loss')
# loss = tf.keras.losses.MeanSquaredLogarithmicError(
#     reduction=losses_utils.ReductionV2.AUTO, name='mean_squared_logarithmic_error')
O = "adam"
opt = "adam" #defult LR=0.001
# decayed_lr = tf.optimizers.schedules.ExponentialDecay(0.001, decay_steps=100000, decay_rate=0.96, staircase=True)
# opt = tf.optimizers.Adam(decayed_lr)

# metrics: #'mean_absolute_error', 'mean_absolute_percentage_error'
metrics = 'mean_absolute_percentage_error'
m = 'MAPE'

num_filters_B1 = 16
num_filters_B2 = 16
# padding = 'same' # 'valid'


save_name = 'networks/COM_'+str(data_len)+'_'+str(validation_split)+'_'+O+'_'+L+'_'+str(epochs)+'_'+target+'_'+A+trial
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

input_shape=(100, 100, 100, 1)
# Use strategy for Multiple GPUs
tf.debugging.set_log_device_placement(True)
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():

  # define model input
  input_image = Input(shape=(100, 100, 100, 1))
  # add inception module
  layer = inception_module(input_image)
  layer = BatchNormalization()(layer)
  layer = Conv3D(16, 2, strides=2, padding='same', use_bias="False")(layer)
  # add Conv3D block
  layer = conv_module(layer)
  layer = BatchNormalization()(layer)
  layer = SpatialDropout3D(0.1)(layer)
  layer = Conv3D(32, 2, strides=2, padding='same', use_bias="False")(layer)
  # FCL part
  layer = Flatten()(layer)
  layer = dense_module(layer)
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

# Evaluate the model on the test data using `evaluate`
# print("Evaluate on test data")
# results = model.evaluate(x_test, y_test, batch_size=batch_size)
# print("test loss(" + L + "), test(" + m + "):", results)
predict_test = model.predict(x_test)
predict_val = model.predict(x_val)
predict_train = model.predict(x_train)


y_test_mD = y_test
y_train_mD = y_train
y_val_mD = y_val
predict_test_mD = predict_test
predict_train_mD = predict_train
predict_val_mD = predict_val
y_train = np.log10(y_train)
predict_train = np.log10(predict_train)
y_test = np.log10(y_test)
predict_test = np.log10(predict_test)

    
y_test_PE = np.zeros((len(predict_test)), dtype="float64")
y_test_mD_PE = np.zeros((len(predict_test)), dtype="float64")
for i in range(0, len(predict_test)):
    y_test_PE[i] = np.abs(predict_test[i]-y_test[i])/y_test[i]
    y_test_mD_PE[i] = np.abs(predict_test_mD[i]-y_test_mD[i])/y_test_mD[i]
MAPE = np.mean(y_test_PE)
MAPE_mD = np.mean(y_test_mD_PE)

UT_90 = np.percentile(y_test_mD_PE, 90)
indices_remove = np.array(np.where(y_test_mD_PE >= UT_90))
y_test_mD_PE_90 = np.delete(y_test_mD_PE, indices_remove)
y_test_mD_90 = np.delete(y_test_mD, indices_remove)
predict_test_mD_90 = np.delete(predict_test_mD, indices_remove)
MAPE_90_mD = np.mean(np.delete(y_test_mD_PE, indices_remove))
MAPE_90 = np.mean(np.delete(y_test_PE, indices_remove))

MAPE_75_mD = np.mean(np.delete(y_test_mD_PE, np.array(np.where(y_test_mD_PE >= np.percentile(y_test_mD_PE, 75)))))
MAPE_50_mD = np.mean(np.delete(y_test_mD_PE, np.array(np.where(y_test_mD_PE >= np.percentile(y_test_mD_PE, 50)))))

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

UT_90 = np.percentile(y_train_mD_PE, 90)
indices_remove = np.array(np.where(y_train_mD_PE >= UT_90))
y_train_PE_90 = np.delete(y_train_PE, indices_remove)
y_train_mD_90 = np.delete(y_train_mD, indices_remove)
y_train_mD_PE_90 = np.delete(y_train_mD_PE, indices_remove)
predict_train_mD_90 = np.delete(predict_train_mD, indices_remove)


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
print("MAPE < 90th percentile [log]:" + str(np.round(MAPE_90*100, 2))+" %")
print("MAPE < 90th percentile [mD]:" + str(np.round(MAPE_90_mD*100, 2))+" %")
time = np.round((time.time() - t)/60,1)
print("Training "+str(epochs)+" epochs completed in: "+str(time)+ " mins")

Y_summary_red = np.delete(Y_summary, indices_remove, axis=0)


# save results
savetxt(save_name+'/results_summary.csv', Y_summary, delimiter=',')
savetxt(save_name+'/results_summary_dir.csv', direction_test, delimiter=',', fmt="%s")
savetxt(save_name+'/results_summary_train.csv', Y_summary_train, delimiter=',')
savetxt(save_name+'/results_summary_dir_train.csv', direction_train, delimiter=',', fmt="%s")

# Plot Predictions

# min = np.min([np.min(predict_test), np.min(y_test), np.min(predict_train), np.min(y_train)])*0.95
# max = np.max([np.max(predict_test), np.max(y_test), np.max(predict_train), np.max(y_train)])*1.05

# min = min([np.min(y_test), np.min(y_train)])*0.9
# max = max([np.max(y_test), np.max(y_train)])*1.1

min = LT
max = UT
plt.plot([min, max], [min, max], c='b', linewidth=1, alpha=0.9)
plt.scatter(y_train_mD, predict_train_mD, c='g', s=1.5, alpha=0.5)
plt.scatter(y_test_mD, predict_test_mD, c='r', s=1.5)
# plt.xscale("log")
# plt.yscale("log")
# plt.xlim(min, max)
# plt.ylim(min, max)
tl1 = "Model Predictions ("+target+")\n"
tl2_1 = "Test Resuts\n"
tl2_2 = "(MAPE, mD): "+str(np.round(MAPE_mD*100, 2))+" %\n"
tl4 = "MAPE < 90th percentile [mD]:" + str(np.round(MAPE_90_mD*100, 2))+" %"
plt.title(tl1+tl2_1+tl2_2+tl4)
# plt.title('Model Predictions ('+target+")\n Test Resuts ("+m+"): " +
#           str(np.round(results[1], 3))+" [log], (MAPE): " +str(np.round(MAPE_mD, 3))+" [D]")
plt.ylabel('Predicted Permeability')
plt.xlabel('True Permeability')
plt.legend(['True', 'Predictions (Train)', 'Predictions (Test)'], loc='upper left')
plt.savefig(save_name+'/predictions_graph_all.png', dpi=300)
# plt.show()
plt.clf()

# Plot Predictions < 90th percentile
plt.plot([min, max], [min, max], c='b', linewidth=1, alpha=0.9)
plt.scatter(y_train_mD_90, predict_train_mD_90, c='g', s=1.5, alpha=0.5)
plt.scatter(y_test_mD_90, predict_test_mD_90, c='r', s=1.5)
# plt.xscale("log")
# plt.yscale("log")
# plt.xlim(min, max)
# plt.ylim(min, max)
tl1 = "Model Predictions ("+target+")\n"
tl2_1 = "Test Resuts\n"
tl2_2 = " (MAPE, mD): "+str(np.round(MAPE_mD*100, 2))+" %\n"
tl4 = "MAPE < 90th percentile [mD]:" + str(np.round(MAPE_90_mD*100, 2))+" %"
plt.title(tl1+tl2_1+tl2_2+tl4)
# plt.title('Model Predictions ('+target+")\n Test Resuts ("+m+"): " +
#           str(np.round(results[1], 3))+" [log], (MAPE): " +str(np.round(MAPE_mD, 3))+" [D]")
plt.ylabel('Predicted Permeability')
plt.xlabel('True Permeability')
plt.legend(['True', 'Predictions (Train)', 'Predictions (Test)'], loc='upper left')
plt.savefig(save_name+'/predictions_graph_90.png', dpi=300)
# plt.show()
plt.clf()




UT_90 = np.percentile(y_test_mD_PE, 90)
UT_75 = np.percentile(y_test_mD_PE, 75)
UT_50 = np.percentile(y_test_mD_PE, 50)
# Plot Predictions Error (All)
max = np.max([np.max(y_test_mD_PE)*1.2, 0.5])
plt.scatter(y_test_mD, y_test_mD_PE, s=1)
plt.scatter(y_train_mD, y_train_mD_PE, s=1, alpha=0.7)
legend1 = plt.legend(['Predictions (Test)', 'Predictions (Train)'], loc='upper left')
plt.axhline(UT_90, color='r', linestyle='dashed', linewidth=1, alpha=0.8)
plt.axhline(UT_75, color='b', linestyle='dashed', linewidth=1, alpha=0.8)
plt.axhline(UT_50, color='g', linestyle='dashed', linewidth=1, alpha=0.8)
legend2 = plt.legend(['90% Percentile', '75% Percentile', '50% Percentile'], loc='upper right')
plt.gca().add_artist(legend1)
plt.ylim(0, max)
tl1 = "Predictions Error Distribution"
plt.title(tl1)
plt.ylabel('error')
plt.xlabel('True Permeability [mD]')
plt.savefig(save_name+'/predictions_error_all_graph.png', dpi=300)
# plt.show()
plt.clf()

# Plot Predictions Error  < 90th percentile
max = np.max([np.max(y_test_mD_PE_90)*1.5, 0.5])
plt.scatter(y_test_mD_90, y_test_mD_PE_90, s=1, label='Predictions (Test)')
plt.scatter(y_train_mD_90, y_train_mD_PE_90,s=1, label='Predictions (Train)', alpha=0.7)
legend1 = plt.legend(loc='upper left')
plt.axhline(UT_90, color='r', linestyle='dashed', linewidth=1, alpha=0.8)
plt.axhline(UT_75, color='b', linestyle='dashed', linewidth=1, alpha=0.8)
plt.axhline(UT_50, color='g', linestyle='dashed', linewidth=1, alpha=0.8)
legend2 = plt.legend(['90% Percentile', '75% Percentile', '50% Percentile'], loc='upper right')
plt.gca().add_artist(legend1)
plt.ylim(0, max)
tl1 = "Predictions Error Distribution < 90th percentile"
plt.title(tl1)
plt.ylabel('error')
plt.xlabel('True Permeability [mD]')
plt.annotate('Mean error= '+str(np.round(MAPE_90_mD, 3)), (np.max(y_test_mD)*0.75,UT_90), textcoords="offset points", xytext=(0,-10), ha='left', size=9, weight='bold')
plt.annotate('Mean error= '+str(np.round(MAPE_75_mD, 3)), (np.max(y_test_mD)*0.75,UT_75), textcoords="offset points", xytext=(0,-10), ha='left', size=9, weight='bold')
plt.annotate('Mean error= '+str(np.round(MAPE_50_mD, 3)), (np.max(y_test_mD)*0.75,UT_50), textcoords="offset points", xytext=(0,-10), ha='left', size=9, weight='bold')
plt.savefig(save_name+'/predictions_error_90_graph.png', dpi=300)
# plt.show()
plt.clf()



# Draw Box Plot
# https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.boxplot.html
plt.boxplot(np.abs(y_test_mD_error), sym='b.', showmeans=True)
tl1 = "Permeability Prediction Error (<90th percentile)\n"
tl2 = "mean absolute error: "+str(int(np.mean(np.abs(y_test_mD_error))))+" [mD]"
plt.title(tl1+tl2)
labels = ['3D CNN']
ticks = [1]
plt.xticks(ticks, labels, rotation='horizontal')
# plt.xlabel('3D CNN')
plt.ylabel('Error [mD]')
plt.savefig(save_name+'/box_plot_mD.png', dpi=300)
# plt.show()
plt.clf()

# Draw Box Plot
# https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.boxplot.html
plt.boxplot(np.abs(y_test_mD_PE_90*100), sym='b.', showmeans=True)
tl1 = "Permeability Prediction Error (<90th percentile)\n"
tl2 = "mean error: "+str(np.round(MAPE_90_mD*100, 2))+" %"
plt.title(tl1+tl2)
labels = ['3D CNN']
ticks = [1]
plt.xticks(ticks, labels, rotation='horizontal')
plt.ylabel('Error (%)')
plt.savefig(save_name+'/box_plot_error_ratio.png', dpi=300)
# plt.show()
plt.clf()



# Visualize data (Histogarm)
for (x,y) in ([y_train_mD, "Train"], [y_test_mD, "Test"], [y_val_mD, "Val"], [y_test_mD_PE*100, "Test Error(%)"], [y_test_mD_PE_90*100, "Test Error (<90th percentile)"] ):
  bins = int(np.round((np.max(x) - np.min(x))/100))
  title = " Samples Histogram, 100 mD step"
  title_2 = "Permeability Value [mD]"
  if bins < 10:
    bins = int(np.round((np.max(x) - np.min(x))/5))
    title = " Histogram"
    title_2 = "Error(%)"
  plt.hist(x, bins=bins, color='b', alpha=0.8)  # arguments are passed to np.histogram
  plt.axvline(x.mean(), color='r', linestyle='dashed', linewidth=1)
  plt.axvline(np.median(x), color='k', linestyle='dashed', linewidth=1)
  plt.ylabel('Count')
  plt.xlabel(title_2)
  plt.legend(['Mean', 'Median'],
             loc='upper right')
  plt.title(str(y)+title)
  plt.savefig(save_name+'/histogram_'+str(LT)+'_'+str(UT)+'_'+str(y)+'_'+str(len(x))+'.png', dpi=300)
  # plt.show()
  plt.clf()
