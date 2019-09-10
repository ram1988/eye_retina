from skimage import transform, io
import csv
import os
import pickle
import tensorflow as tf
import numpy as np
from cnn_classifier import *
import scipy
import itertools
import sys


vector_size = 500
num_classes = 5
batch_size = 1000

def indices_to_one_hot(data, nb_classes=2):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


def serving_input_receiver_fn():
    feature_spec = {
        'image': tf.FixedLenFeature([], dtype=tf.string)
    }

    serialized_tf_example = tf.placeholder(
        dtype=tf.string, shape=[1],
        name='input_image_tensor')

    received_tensors = {'images': serialized_tf_example}
    features = tf.parse_example(serialized_tf_example, feature_spec)

    fn = lambda image: parse_feature_label(image, is_predict=True)
    features['image'] = tf.map_fn(fn, features['image'], dtype=tf.float32)

    return tf.estimator.export.ServingInputReceiver(features, received_tensors)


cnnclassifier = CNNClassifier(vector_size, num_classes)
model = cnnclassifier.get_classifier_model()


def parse_feature_label(filename, label=None, is_predict=False):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string)
    image_decoded = tf.image.convert_image_dtype(
        image_decoded, dtype=tf.float64)
    image_resized = tf.image.resize_images(
        image_decoded, (vector_size, vector_size))

    if is_predict:
        return image_resized
    else:
        return (image_resized, label)


def prepare_image_files(path, is_training=False):
    image_records = []
    image_labels = []
    ids = []
    i = 0
    with open(path) as train_labels:
        csv_reader = csv.reader(train_labels)
        for row in csv_reader:
            if is_training:
                label = int(row[1])
                image_labels.append(label)
                image_file = "train_images/"+row[0]+".png"
            else:
                image_file = "test_images/"+row[0]+".png"
            image_records.append(image_file)
            ids.append(row[0])
            i = i+1
    print(len(image_records))

    return image_records, image_labels, ids



def train(train_set, train_labels):
    def train_input_fn(features, labels, batch_size, repeat_count):
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.shuffle(buffer_size=300)
        dataset = dataset.map(
            lambda x, y: parse_feature_label(x, y)).batch(batch_size)
        dataset = dataset.repeat(1).prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
        return dataset

    train_image_features = train_set
    train_image_labels = train_labels
    train_image_labels = np.array(train_image_labels)
    train_image_labels = indices_to_one_hot(train_image_labels, num_classes)
    train_image_labels = np.reshape(train_image_labels, (-1, num_classes))

    steps = (len(train_image_features)/batch_size)-1
    steps = steps if steps > 0 else 1

    # Train the Model in loop in future
    model.train(input_fn=lambda: train_input_fn(
        train_image_features, train_image_labels, 100, 20), steps=steps)


    eval_len = int(0.2*len(train_set))
    return evaluate(train_set[:eval_len],train_labels[:eval_len])


def evaluate(val_set, val_labels):
    def eval_input_fn(features, labels, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.shuffle(buffer_size=300)
        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(
                lambda x, y: parse_feature_label(x, y),
                batch_size=1,
                num_parallel_batches=1,
                drop_remainder=False))
        dataset = dataset.repeat(1).prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
        return dataset

    val_image_features = val_set
    val_image_labels = val_labels
    val_image_labels = np.array(val_image_labels)
    val_image_labels = indices_to_one_hot(val_image_labels, num_classes)
    val_image_labels = np.reshape(val_image_labels, (-1, num_classes))

    steps = (len(val_image_features) / batch_size) - 1
    steps = steps if steps > 0 else 1

    # Train the Model
    evaluate_result = model.evaluate(input_fn=lambda: eval_input_fn(
        val_image_features, val_image_labels, 100), steps=steps)
    print("Evaluation results")
    for key in evaluate_result:
        print("   {}, was: {}".format(key, evaluate_result[key]))

    persisted_path = model.export_saved_model(
        "predictor", serving_input_receiver_fn=serving_input_receiver_fn)

    return persisted_path


def predict(model_path):
    test_images, lbls, ids = prepare_image_files("test.csv")
    print(test_images[0])

    output = []
    predictor = tf.contrib.predictor.from_saved_model(model_path)
    for tst_img in test_images:
        content_tf_list = tf.train.BytesList(value=[tst_img.encode()])
        example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'image': tf.train.Feature(
                                bytes_list=content_tf_list
                            )
                        }
                    )
                )
        serialized_example = example.SerializeToString()
        print(serialized_example)
        op = predictor({'images': [serialized_example]})["classes"][0]
        print(op)
        output.append(op)
    write_results(output,ids)

def write_results(predict_results,ids):
   idx = 0
   f = open("sample_submission1.csv", "w+")
   for prediction in predict_results:
       # print(prediction)
       # print(idx)
       if idx<len(ids):
            f.write(ids[idx]+","+str(prediction)+"\n")
       idx = idx+1
   f.close()

args = sys.argv
mode = args[1]
if mode == "tp":
    train_set,train_labels,ids = prepare_image_files("train.csv",True)
    model_path = train(train_set,train_labels)
    predict(model_path)
elif mode == "t":
    train_set,train_labels,ids = prepare_image_files("train.csv",True)
    model_path = train(train_set,train_labels)
    print("Model path...")
    print(model_path)
elif mode == "p":
    model_path = args[2]
    predict(model_path)
else:
    print("invalid mode... please enter t, p or tp")

