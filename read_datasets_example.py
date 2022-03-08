# tensorflow 2.0+

import os

import cv2
import tensorflow as tf
import numpy as np


def _parse_function(example_proto):
    parsed_features = tf.io.parse_single_example(example_proto,
                                                 features = {'label': tf.io.FixedLenFeature([], tf.int64),
                                                             'image': tf.io.FixedLenFeature([], tf.string)
                                                             })
    image = tf.io.decode_raw(parsed_features['image'], tf.float32)
    image = tf.reshape(image, [224, 224, 3])

    label = tf.cast(parsed_features['label'], tf.int32)

    return image, label


def readTFRcord(datafolder, batch_size):
    files = tf.io.match_filenames_once(datafolder)
    print(files)
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(20)
    dataset = dataset.batch(batch_size)

    return dataset


if __name__ == '__main__':
    datafolder = r"/Chinese Spirits Bubble Datasets" # Path to dataset folder
    print(datafolder)
    dataset = readTFRcord(datafolder = os.path.join(datafolder, '*.tfrecord'), batch_size = 8)

    for feature, label in dataset:
        img_batch, lab_bacth = feature.numpy(), label.numpy()
        for img, lab in zip(img_batch, lab_bacth):
            img = np.array(img, dtype = np.uint8)
            cv2.imshow('image', img)
            print(img.shape, lab)
            cv2.waitKey(1000)
