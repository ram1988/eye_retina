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

#https://medium.com/tensorflow/how-to-write-a-custom-estimator-model-for-the-cloud-tpu-7d8bd9068c26
def prepare_test_data(image_data):
     parsed = tf.parse_example(image_data, {
      'image/encoded': tf.FixedLenFeature((), tf.string, '')    
      })
     image = parse_feature_label(parsed['image/encoded'], is_predict=True)
     print("parsed dart-->"+str(image))
     return image
    
def serving_input_receiver_fn():    
    feature_spec = {
        'image': tf.FixedLenFeature([], dtype=tf.string)
    }
    
    serialized_tf_example = tf.placeholder(
        dtype=tf.string, shape=[1], 
        name='input_image_tensor')
    
    received_tensors = { 'images': serialized_tf_example }
    features = tf.parse_example(serialized_tf_example, feature_spec)
    
    fn = lambda image: parse_feature_label(image, is_predict=True)
    features['image'] = tf.map_fn(fn, features['image'], dtype=tf.float32)

    return tf.estimator.export.ServingInputReceiver(features, received_tensors)



cnnclassifier = CNNClassifier(vector_size,num_classes)
model = cnnclassifier.get_classifier_model()


def parse_feature_label(filename,label=None,is_predict=False):
    print("read image-->")
    print(filename)
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [vector_size, vector_size])
    print(image_resized.shape)

    if is_predict:
        return image_resized
    else:
        return image_resized,label

def prepare_image_files(path,is_training=False):
    image_records = []
    image_labels = []
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
            i = i+1
            print(i)
    print(len(image_records))

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

# input_fn for evaluation and predicitions (labels can be null)
def pred_input_fn(features,  batch_size=1):
    print("predict labels.....")
    dataset = tf.data.Dataset.from_tensor_slices((features))
    dataset = dataset.shuffle(buffer_size=300)
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            lambda x: parse_feature_label(x,is_predict=True),
            batch_size=1000,
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
    model.train(input_fn=lambda:train_input_fn(train_image_features,train_image_labels,100,20),steps = steps)
    print(model)
    model.export_saved_model("predictor",serving_input_receiver_fn=serving_input_receiver_fn)

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

def print_rec(record):
    print("print---r")
    print(record)
    return record

#https://stackoverflow.com/questions/48904313/invalidargumenterror-when-loading-tfrecord-file
def predict():
   test_images,lbls = prepare_image_files("test.csv")
   print(test_images[0])

   predictor = tf.contrib.predictor.from_saved_model("predictor/1566771866")
   for tst_img in test_images:
        content_tf_list = tf.train.BytesList(value=[str.encode(tst_img)])
        example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'image': tf.train.Feature(
                                bytes_list=content_tf_list
                            )
                        }
                    )
                )
        print(example)
        serialized_example = example.SerializeToString()
        print(serialized_example)
        op = predictor({'images': [serialized_example]})
        print(op)

def predictors():
   test_images,lbls = prepare_image_files("test.csv")
   print(len(test_images))
   #print(test_images)

   predict_results = model.predict(input_fn=lambda:pred_input_fn(test_images))
   print(predict_results)
   overall_results = []
   for prediction in predict_results:
       overall_results.append(prediction["classes"])
       print(prediction["classes"])

   return overall_results

#train_set,train_labels = prepare_image_files("train.csv",True)
#train(train_set,train_labels)
predict()

