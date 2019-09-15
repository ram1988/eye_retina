import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

#https://medium.com/@utsumuki_neko/using-inception-v3-from-tensorflow-hub-for-transfer-learning-a931ff884526
#https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/3
class ResnetClassifier:

	def __init__(self,vector_size, img_classes):
		self.vector_size = vector_size
		self.img_classes = img_classes


	def cnn_model_fn(self,features, labels, mode):
		"""Model function for CNN."""
		# Input Layer
		if isinstance(features, dict):
			features = features['image']   
		input_layer = tf.reshape(features, [-1, self.vector_size, self.vector_size, 3])

		model = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/3")
		print("resnet model---->")
		print(model)
		model = model(input_layer)
		# Dense Layer
		pool2_flat = tf.layers.flatten(model)
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