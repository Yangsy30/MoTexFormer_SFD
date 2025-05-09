import os
from pathlib import Path
import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod
import matplotlib.pyplot as plt
import glob
import random
import numpy as np
from math import ceil
from patchify import patchify, unpatchify
from numpy.core.fromnumeric import argmax
from PIL import Image
from model import mtf
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte
import cv2

# tf.compat.v1.disable_eager_execution()

random.seed(12)
series_num = 1  
band_num = 16
additional_channels = 3  
total_channels = band_num + additional_channels
OUTPUT_CHANNELS = 5  
input_h = 512
input_w = 512
train_rate = 0.8



def decode_npy(data_path):


    img_label_data = np.load(data_path.numpy())  
    img_label_data = cv2.resize(img_label_data,(input_h, input_w),interpolation=cv2.INTER_NEAREST)
    selected_channels = [3, 4, 6, 12, 16, 17, 18, 19, 20] 
    image = img_label_data[:, :, selected_channels]  
    label = img_label_data[:, :, -1]  
    image = image.astype(np.float32)

    filename = data_path.numpy().decode('utf-8')

    threshold_channels = [0, 1, 2, 3]
    for ch in threshold_channels:
        image[:, :, ch] = np.clip(image[:, :, ch], np.min(thresholds[ch]), np.max(thresholds[ch]))

    normalize_channels = [4, 5, 6, 7]
    for ch in normalize_channels:
        image[:, :, ch] = image[:, :, ch] / 255.0

    image[:, :, -1] = np.where(image[:, :, -1] != 0, 1, 0) 
    return image, label


def process_path(data_path):
    image, mask = tf.py_function(decode_npy, [data_path], [tf.float32, tf.uint8])
    image.set_shape([input_h, input_w, 9])
    mask.set_shape([input_h, input_w])
    return image, mask


if __name__ == '__main__':

    model_name = 'XXX'

    raw_data_path = Path('')  #

    data_list = list(glob.glob(""))  
    fog_dates = [os.path.basename(l)[0:8] for l in data_list]
    unique_dates = list(set(fog_dates)) 
    unique_dates.sort() 


    random.shuffle(unique_dates) 
    training_dates = unique_dates[:int(train_rate * len(unique_dates))] 
    test_dates = unique_dates[int(train_rate * len(unique_dates)):] 

    label_train = []
    label_valid = []
    for file in data_list:
        file_date = os.path.basename(file)[0:8] 
        if file_date in test_dates: 
            label_valid.append(file) 
        else:
            label_train.append(file) 


    random.shuffle(label_train)
    train_count = len(label_train)
    valid_count = len(label_valid)

    TRAIN_LENGTH = train_count
    TEST_LENGTH = valid_count
    BATCH_SIZE = 2
    BUFFER_SIZE = 1000
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
    EPOCHS = 100
    VAL_SUBSPLITS = 5
    VALIDATION_STEPS = TEST_LENGTH // BATCH_SIZE // VAL_SUBSPLITS
    VALIDATION_STEPS = 1


    ds_train = tf.data.Dataset.from_tensor_slices(label_train)
    ds_train = ds_train.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.batch(BATCH_SIZE).repeat() 
    ds_train = ds_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    ds_test = tf.data.Dataset.from_tensor_slices(label_valid)
    ds_test = ds_test.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.cache().batch(BATCH_SIZE).repeat()
    ds_test = ds_test.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


    model = mtf() 
    model.build(input_shape=(None, 512, 512, 9))

    model.compile(optimizer='adam', loss=sea_fog_loss, metrics=["acc"])
    
    model.summary()

    tf.keras.utils.plot_model(model, to_file=f'{model_name}.png', show_shapes=True)
    
    csv_logger = tf.keras.callbacks.CSVLogger(f'{model_name}_training_log.csv', append=True)

    model_history = model.fit(ds_train, epochs=EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH*2,
                              validation_steps=VALIDATION_STEPS,
                              validation_data=ds_test,
                              callbacks=[csv_logger])
    
    model.save_weights(f'{model_name}_weights.h5')




