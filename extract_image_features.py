from skimage import transform, io
import csv
import os, pickle
import tensorflow as tf
import numpy as np
from cnn_classifier import *
import scipy
import itertools

vector_size=200
num_classes = 5
batch_size = 100

# Feature extractor
def extract_features(image_path, vector_size=200):
    image = io.imread(image_path, as_gray=True)
    image = transform.resize(image,(vector_size,vector_size),mode='symmetric',preserve_range=True)
    return image

def indices_to_one_hot(data, nb_classes=2):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def prepare_image_set(path,file_name):

    with open(path) as train_labels:
        csv_reader = csv.reader(train_labels)
        i = 0
        batch_num = 1
        image_train_labels = []
        for row in csv_reader:
            image_files = os.listdir(row[0])
            label = row[1]
            for image in image_files:
                image_file = row[0]+image
                image_features = extract_features(image_file)
                image_train_labels.append((image_features,label))
                i = i+1
                print(i)
                if len(image_train_labels) == 5000:
                    print("written")
                    pickle.dump(image_train_labels, open(file_name+str(batch_num)+".pkl", 'wb'))
                    batch_num = batch_num+1
                    image_train_labels = []

        pickle.dump(image_train_labels, open(file_name + str(batch_num) + ".pkl", 'wb'))

    return image_train_labels

def serving_input_rvr_fn():
    serialized_tf_example = tf.placeholder(dtype=tf.string, shape=[batch_size], name='input_tensors')
    receiver_tensors = {"predictor_inputs": serialized_tf_example}
    feature_spec ={"images": tf.FixedLenFeature([vector_size,vector_size], tf.float32)}
    features = tf.parse_example(serialized_tf_example, feature_spec)
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


cnnclassifier = CNNClassifier(vector_size,num_classes)
model = cnnclassifier.get_classifier_model()


def parse_feature_label(filename,label):
    print("read image")
    print(filename)
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [vector_size, vector_size])
    print(image_resized.shape)
    print(label.shape)
    return image_resized,label

def prepare_image_files(path):
    image_records = []
    image_labels = []
    i = 0
    with open(path) as train_labels:
        csv_reader = csv.reader(train_labels)
        for row in csv_reader:
            label = int(row[1])
            image_file = "train_images/"+row[0]+".png"
            image_records.append(image_file)
            image_labels.append(label)
            i = i+1
            print(i)
    print(len(image_records))
    print(image_labels)

    return image_records,image_labels



#https://medium.com/@vincentteyssier/tensorflow-estimator-tutorial-on-real-life-data-aa0fca773bb
#https://github.com/marco-willi/tf-estimator-cnn/blob/master/estimator.py
#https://www.tensorflow.org/guide/datasets#consuming_numpy_arrays
#change the logic accordingly
def train_input_fn(features, labels, batch_size, repeat_count):
    print("train labels.....")
    print(labels)
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

# input_fn for evaluation and predicitions (labels can be null)
def eval_input_fn(features, labels, batch_size):
    print("eval labels.....")
    print(labels)
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

def train(train_set,train_labels):
    train_image_features = train_set
    train_image_labels = train_labels
    train_image_labels = np.array(train_image_labels)
    train_image_labels = indices_to_one_hot(train_image_labels,5)
    train_image_labels = np.reshape(train_image_labels,(-1,5))
    print("shapes")
    print(len(train_image_features))
    print(len(train_image_labels))

    steps = (len(train_image_features)/batch_size)-1
    steps = steps if steps>0  else 1

    # Train the Model
    print("dddd")
    model.train(input_fn=lambda:train_input_fn(train_image_features,train_image_labels,100,20),steps = steps)
    print(model)

def evaluate(val_set,val_labels):
    val_image_features = val_set
    val_image_labels = val_labels
    print("evaluate...")
    print(len(val_image_features))
    val_image_labels = np.array(val_image_labels)
    val_image_labels = indices_to_one_hot(val_image_labels,5)
    val_image_labels = np.reshape(val_image_labels, (-1, 2))

    #val_image_labels = np.array(val_image_labels)
    #val_image_labels = np.reshape(val_image_labels, (-1, 2))

    steps = (len(val_image_features) / batch_size) - 1
    steps = steps if steps > 0  else 1

    # Train the Model
    evaluate_result = model.evaluate(input_fn=lambda:eval_input_fn(val_image_features,val_image_labels,100), steps=steps)
    print ("Evaluation results")
    for key in evaluate_result:
        print("   {}, was: {}".format(key, evaluate_result[key]))
    #model.export_savedmodel("test_model",serving_input_receiver_fn=serving_input_rvr_fn)


def predict():
   pred_image_features = train_image_features_glb[train_size:][0:3]
   #pred_image_features = np.array(pred_image_features)
   print("3")
   print(len(pred_image_features))
   print(pred_image_features[0].shape)
   model_input = tf.train.Example(features=tf.train.Features(feature={"images":tf.train.Feature(float_list=tf.train.FloatList(value=pred_image_features[0]))}))
   model_input = model_input.SerializeToString()
   predictor = tf.contrib.predictor.from_saved_model("test_model/1545839240")
   output_dict = predictor({"predictor_inputs": [model_input]})
   print(output_dict)


train_set,train_labels = prepare_image_files("train.csv")
#val_set,val_labels = prepare_image_files("test.csv")

train(train_set,train_labels)
#evaluate(val_set,val_labels)
#predict()
