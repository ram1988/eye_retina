import tensorflow as tf
import numpy as np

#https://towardsdatascience.com/first-contact-with-tensorflow-estimator-69a5e072998d
class CNNClassifier:

	def __init__(self,vector_size, img_classes):
		self.vector_size = vector_size
		self.img_classes = img_classes


	def cnn_model_fn(self,features, labels, mode):
		"""Model function for CNN."""
		# Input Layer
		if isinstance(features, dict):
			features = features['image']   
		input_layer = tf.reshape(features, [-1, self.vector_size, self.vector_size, 3])

		# Convolutional Layer #1
		conv1 = tf.layers.conv2d(
			inputs=input_layer,
			filters=10,
			kernel_size=[5, 5],
			padding="same",
			activation=tf.nn.relu)

		# Pooling Layer #1
		pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)

		# Convolutional Layer #1
		conv2 = tf.layers.conv2d(
			inputs=pool1,
			filters=20,
			kernel_size=[4, 4],
			padding="same",
			activation=tf.nn.relu)

		# Pooling Layer #1
		pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)

		# Convolutional Layer #1
		conv3 = tf.layers.conv2d(
			inputs=pool2,
			filters=30,
			kernel_size=[3, 3],
			padding="same",
			activation=tf.nn.relu)

		# Pooling Layer #1
		pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[3, 3], strides=2)

		# Convolutional Layer #1
		conv4 = tf.layers.conv2d(
			inputs=pool3,
			filters=40,
			kernel_size=[2, 2],
			padding="same",
			activation=tf.nn.relu)

		# Pooling Layer #1
		pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[3, 3], strides=2)


		# Dense Layer
		pool2_flat = tf.layers.flatten(pool4)
		dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
		dropout = tf.layers.dropout(
			inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

		# Logits Layer
		logits = tf.layers.dense(inputs=dropout, units=5)

		predictions = {
			# Generate predictions (for PREDICT and EVAL mode)
			"classes": tf.argmax(input=logits, axis=1),
			# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
			# `logging_hook`.
			"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
		}

		if mode == tf.estimator.ModeKeys.PREDICT:
			return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

		# Calculate Loss (for both TRAIN and EVAL modes)
		loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

		# Configure the Training Op (for TRAIN mode)
		if mode == tf.estimator.ModeKeys.TRAIN:
			optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
			train_op = optimizer.minimize(
				loss=loss,
				global_step=tf.train.get_global_step())
			return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

		# Add evaluation metrics (for EVAL mode)
		eval_metric_ops = {
			"accuracy": tf.metrics.accuracy(
				labels=tf.argmax(input=labels, axis=1), predictions=predictions["classes"])
		}
		return tf.estimator.EstimatorSpec(
			mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

	def get_classifier_model(self):
		print("get the model...")
		return tf.estimator.Estimator(
			model_fn = self.cnn_model_fn)