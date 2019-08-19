import tensorflow as tf
import numpy as np

#https://towardsdatascience.com/first-contact-with-tensorflow-estimator-69a5e072998d
class CNNClassifier:

	def __init__(self,vector_size, img_classes):
		self.vector_size = vector_size
		self.img_classes = img_classes


	def define_model_net(self,img_features):
		# Convolutional Layer #1
		img_features = tf.cast(img_features, tf.float32)
		conv1 = tf.layers.conv2d(
			inputs=img_features,
			filters=5,
			kernel_size=[3, 3],
			padding="same",
			activation=tf.nn.relu)
		# Pooling Layer #1
		pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=1)
		print("pool1 shape....")
		print(pool1.shape)

		# Convolutional Layer #2 and Pooling Layer #2
		conv2 = tf.layers.conv2d(
			inputs=pool1,
			filters=10,
			kernel_size=[3, 3],
			padding="same",
			activation=tf.nn.relu)
		pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

		print("pool2 shape....")
		print(pool2.shape)

		conv3 = tf.layers.conv2d(
			inputs=pool2,
			filters=15,
			kernel_size=[3, 3],
			padding="same",
			activation=tf.nn.relu)
		pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

		print("pool3 shape....")
		print(pool3.shape)

		conv4 = tf.layers.conv2d(
			inputs=pool3,
			filters=20,
			kernel_size=[3, 3],
			padding="same",
			activation=tf.nn.relu)
		pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

		print("pool4 shape....")
		print(pool4.shape)

		# Dense Layer
		pool4_flat = tf.reshape(pool4, [-1, 24 * 24 * 20])
		dense = tf.layers.dense(inputs=pool4_flat, units=1024, activation=tf.nn.relu)
		dropout = tf.layers.dropout(
			inputs=dense, rate=0.4, training=True)

		print("return model net")

		# Logits Layer
		return tf.layers.dense(inputs=dropout, units=self.img_classes)



	def __model_fn(self,features, labels, mode, params):
		image_features = features
		print("MODEL@@@@@@")
		print(image_features)
		print(labels)
		#image_features = features
		img_features = tf.reshape(image_features, [-1, self.vector_size, self.vector_size, 1])
		self.logits = self.define_model_net(img_features)
		if mode == tf.estimator.ModeKeys.TRAIN:
			loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=self.logits)
			optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
			train_op = optimizer.minimize(
				loss=loss,
				global_step=tf.train.get_or_create_global_step())
			return self.__train_model_fn(labels, mode, params, self.logits, loss, train_op)
		elif mode == tf.estimator.ModeKeys.EVAL:
			print("evaluate...111")
			obj = self.__eval_model_fn(labels,self.logits,loss)
			print("val ends")
			return obj
		else:
			return self.__predict_model_fn(self.logits)

	def __train_model_fn(self,image_labels,mode,params,logits,loss,train_op):
		print(mode)
		print("training....")
		print(image_labels)
		image_labels = tf.cast(image_labels, tf.float32)
		print(image_labels.shape)
		print(tf.size(image_labels))

		#loss = tf.losses.softmax_cross_entropy(onehot_labels=image_labels, logits=logits)
		# Configure the Training Op (for TRAIN mode)

		return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op)


	def __eval_model_fn(self,image_labels,logits,loss):
		image_labels = tf.cast(image_labels, tf.float32)
		print("eval model...")
		print(logits)
		#loss = tf.losses.softmax_cross_entropy(onehot_labels=image_labels, logits=logits)
		# Add evaluation metrics (for EVAL mode)
		predictions = {
			# Generate predictions (for PREDICT and EVAL mode)
			"classes": tf.argmax(input=logits, axis=1),
			# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
			# `logging_hook`.
			"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
		}
		print("predictions....")
		eval_metric_ops = {
				"accuracy": tf.metrics.accuracy(
					labels=tf.argmax(input=image_labels, axis=1), predictions=predictions["classes"])}
		print(image_labels)
		return tf.estimator.EstimatorSpec(
				mode=tf.estimator.ModeKeys.EVAL, loss=loss, eval_metric_ops=eval_metric_ops)

	def __predict_model_fn(self,logits):
		print("PRED....")
		print(logits)
		probabilities = tf.nn.softmax(logits)
		class_int = tf.argmax(logits, axis=1)
		predictions = {
				"classes": class_int,
				"probabilities": probabilities
			}
		return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT,
										  predictions=predictions,
										  export_outputs={
											  'classify': tf.estimator.export.PredictOutput(predictions)
										  })


	def get_classifier_model(self):
		print("get the model...")
		return tf.estimator.Estimator(
			model_fn = self.__model_fn)