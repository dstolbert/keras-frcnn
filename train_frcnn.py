import sys
import time
import random
import pprint
import numpy as np
from math import inf
from typing import Any
from scipy.sparse import construct
from sklearn.model_selection import train_test_split

# TF
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Progbar
from tensorflow.keras.optimizers import Adam

# Internal
from keras_frcnn import config, data_generators
from keras_frcnn import losses as loss_funcs
import keras_frcnn.roi_helpers as roi_helpers
from keras_frcnn.simple_parser import get_data

# Limit GPU access if
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 8GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8*1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


sys.setrecursionlimit(40000)

# pass the settings from the command line, and persist them in the config object
C = config.Config("./config.json")

# Ensure we have a train path specified
if not C.train_path:
	print(f"Invalid training path specified {C.train_path}")
	raise ValueError

# load model
nn: Any = None
if C.network == "vgg":
	from keras_frcnn import vgg
	nn = vgg
elif C.network == "resnet":
	from keras_frcnn import resnet
	nn = resnet
else:
	print(f"Not a valid model: {C.network}")
	raise ValueError

all_imgs, classes_count, class_mapping = get_data(C.train_path)
train_imgs, val_imgs = train_test_split(all_imgs, test_size=0.2, random_state=C.seed)

if 'bg' not in classes_count:
	classes_count['bg'] = 0
	class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

print(class_mapping)

inv_map = {v: k for k, v in class_mapping.items()}

print('Training images per class:')
pprint.pprint(classes_count)
print(f'Num classes (including bg) = {len(classes_count)}')

random.shuffle(train_imgs)

num_imgs = len(train_imgs)

print(f'Num train samples {len(train_imgs)}')
print(f'Num val samples {len(val_imgs)}')

data_gen_train = data_generators.get_anchor_gt(
	train_imgs, classes_count, C, 
	nn.get_img_output_length, "tf",
	mode='train'
)
data_gen_val = data_generators.get_anchor_gt(
	val_imgs, classes_count, C, 
	nn.get_img_output_length,"tf", 
	mode='val'
)

input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)

# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

try:
	print(f'loading weights from {C.base_net_weights}')
	model_rpn.load_weights(C.base_net_weights, by_name=True)
	model_classifier.load_weights(C.base_net_weights, by_name=True)
except:
	print('Could not load pretrained model weights. Weights can be found in the keras application folder \
		https://github.com/fchollet/keras/tree/master/keras/applications')

rpn_optimizer = Adam(learning_rate=1e-5)
class_optimizer = Adam(learning_rate=1e-5)

# RPN losses
rpn_cls_loss = loss_funcs.rpn_loss_cls(num_anchors)
rpn_regr_loss = loss_funcs.rpn_loss_regr(num_anchors)

# Class losses
class_cls_loss = loss_funcs.class_loss_cls
class_regr_loss = loss_funcs.class_loss_regr(len(classes_count)-1)

model_all.compile(optimizer='sgd', loss='mae')

epoch_length = C.epoch_length
num_epochs = C.epochs
iter_num = 0

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf

class_mapping_inv = {v: k for k, v in class_mapping.items()}
print('Starting training')

# METRICS
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
train_precision_metric = tf.keras.metrics.Precision()
train_recall_metric = tf.keras.metrics.Recall()

val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_precision_metric = tf.keras.metrics.Precision()
val_recall_metric = tf.keras.metrics.Recall()


def train_step(X, Y, img_data):

	Y = [tf.cast(Y[0], 'float32'), tf.cast(Y[1], 'float32')]

	# Train the rpn model
	with tf.GradientTape() as tape:
		P_rpn = model_rpn(X)
		loss_rpn_0 = rpn_cls_loss(Y[0], P_rpn[0])
		loss_rpn_1 = rpn_regr_loss(Y[1], P_rpn[1])
	loss_rpn = [loss_rpn_0, loss_rpn_1]
	grads = tape.gradient([loss_rpn_0, loss_rpn_1], model_rpn.trainable_weights)
	rpn_optimizer.apply_gradients(zip(grads, model_rpn.trainable_weights))

	R = roi_helpers.rpn_to_roi(P_rpn[0].numpy(), P_rpn[1].numpy(), C, "tf", use_regr=True, overlap_thresh=0.7, max_boxes=300)
	# note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
	X2, Y1, Y2, ious = roi_helpers.calc_iou(R, img_data, C, class_mapping)

	if X2 is None:
		rpn_accuracy_rpn_monitor.append(0)
		rpn_accuracy_for_epoch.append(0)
		return loss_rpn, [], []

	neg_samples = np.where(Y1[0, :, -1] == 1)
	pos_samples = np.where(Y1[0, :, -1] == 0)

	if len(neg_samples) > 0:
		neg_samples = neg_samples[0]
	else:
		neg_samples = []

	if len(pos_samples) > 0:
		pos_samples = pos_samples[0]
	else:
		pos_samples = []
	
	rpn_accuracy_rpn_monitor.append(len(pos_samples))
	rpn_accuracy_for_epoch.append((len(pos_samples)))

	if C.num_rois > 1:
		if len(pos_samples) < C.num_rois//2:
			selected_pos_samples = pos_samples.tolist()
		else:
			selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
		try:
			selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
		except:
			selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()

		sel_samples = selected_pos_samples + selected_neg_samples
	else:
		# in the extreme case where num_rois = 1, we pick a random pos or neg sample
		selected_pos_samples = pos_samples.tolist()
		selected_neg_samples = neg_samples.tolist()
		if np.random.randint(0, 2):
			sel_samples = random.choice(neg_samples)
		else:
			sel_samples = random.choice(pos_samples)

	# Train the classification model
	with tf.GradientTape() as tape:
		P_class = model_classifier([X, X2[:, sel_samples, :]])
		loss_class_0 = class_cls_loss(Y1[:, sel_samples, :], P_class[0]) # classifiation loss
		loss_class_1 = class_regr_loss(Y2[:, sel_samples, :], P_class[1]) # regression loss
	
	# Update classification accuracy metric
	train_acc_metric.update_state(P_class[0], Y1[:, sel_samples, :])
	train_precision_metric.update_state(P_class[0], Y1[:, sel_samples, :])
	train_recall_metric.update_state(P_class[0], Y1[:, sel_samples, :])

	class_metrics = [train_acc_metric.result(), train_precision_metric.result(), train_recall_metric.result()]
	loss_class = [loss_class_0, loss_class_1]
	grads = tape.gradient([loss_class_0, loss_class_1], model_classifier.trainable_weights)
	class_optimizer.apply_gradients(zip(grads, model_classifier.trainable_weights))

	return loss_rpn, loss_class, class_metrics


# Get validation accuracy after each epoch
def computeValidation():

	for _ in enumerate(val_imgs):

		X, Y, img_data = next(data_gen_val)
		
		# RPN
		P_rpn = model_rpn(X)

		R = roi_helpers.rpn_to_roi(P_rpn[0].numpy(), P_rpn[1].numpy(), C, "tf", use_regr=True, overlap_thresh=0.7, max_boxes=300)
		# note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
		X2, Y1, _, _ = roi_helpers.calc_iou(R, img_data, C, class_mapping)

		if X2 is None:
			neg_samples = (np.array([]),)
			pos_samples = (np.array([]),)
			continue
		else:
			neg_samples = np.where(Y1[0, :, -1] == 1)
			pos_samples = np.where(Y1[0, :, -1] == 0)

		if len(neg_samples) > 0:
			neg_samples = neg_samples[0]
		else:
			neg_samples = []

		if len(pos_samples) > 0:
			pos_samples = pos_samples[0]
		else:
			pos_samples = []

		if C.num_rois > 1:
			if len(pos_samples) < C.num_rois//2:
				selected_pos_samples = pos_samples.tolist()
			else:
				selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
			try:
				selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
			except:
				selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()

			sel_samples = selected_pos_samples + selected_neg_samples
		else:
			# in the extreme case where num_rois = 1, we pick a random pos or neg sample
			selected_pos_samples = pos_samples.tolist()
			selected_neg_samples = neg_samples.tolist()
			if np.random.randint(0, 2):
				sel_samples = random.choice(neg_samples)
			else:
				sel_samples = random.choice(pos_samples)

		# Classification model
		P_class = model_classifier([X, X2[:, sel_samples, :]])
		
		# Update classification accuracy metric
		val_acc_metric.update_state(P_class[0], Y1[:, sel_samples, :])
		val_precision_metric.update_state(P_class[0], Y1[:, sel_samples, :])
		val_recall_metric.update_state(P_class[0], Y1[:, sel_samples, :])

	# Get result and reset metric
	precision = val_precision_metric.result()
	recall = val_recall_metric.result()
	f1 = 2 * ( (precision * recall) / (precision + recall) )
	res = (val_acc_metric.result(), val_precision_metric.result(), val_recall_metric.result(), f1)
	val_acc_metric.reset_states()
	val_precision_metric.reset_states()
	val_recall_metric.reset_states()

	return res

best_loss = inf
# TRAINING LOOP
for epoch_num in range(num_epochs):

	progbar = Progbar(epoch_length)
	print(f'\nEpoch {epoch_num + 1}/{num_epochs}')

	while True:

		# try:

			if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
				mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
				rpn_accuracy_rpn_monitor = []
				print(f'\nAverage number of overlapping bounding boxes from RPN = {mean_overlapping_bboxes} for {epoch_length} previous iterations')
				if mean_overlapping_bboxes == 0:
					print('\nRPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')


			X, Y, img_data = next(data_gen_train)
			loss_rpn, loss_class, class_metrics = train_step(X, Y, img_data)

			if len(loss_class) == 0:
				continue

			losses[iter_num, 0] = loss_rpn[0]
			losses[iter_num, 1] = loss_rpn[1]

			losses[iter_num, 2] = loss_class[0]
			losses[iter_num, 3] = loss_class[1]

			progbar.update(iter_num+1, [('rpn_cls', losses[iter_num, 0]), ('rpn_regr', losses[iter_num, 1]),
										('detector_cls', losses[iter_num, 2]), ('detector_regr', losses[iter_num, 3])])

			iter_num += 1
			
			if iter_num == epoch_length:
				# Reset metric at the end of each epoch
				train_acc_metric.reset_states()

				loss_rpn_cls = np.mean(losses[:, 0])
				loss_rpn_regr = np.mean(losses[:, 1])
				loss_class_cls = np.mean(losses[:, 2])
				loss_class_regr = np.mean(losses[:, 3])
				class_acc = np.mean(class_metrics[0])
				class_precision = np.mean(class_metrics[1])
				class_recall = np.mean(class_metrics[2])
				val_acc, val_precision, val_recall, val_f1 = computeValidation()

				mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
				rpn_accuracy_for_epoch = []

				if C.verbose:
					print(f'\nMean number of bounding boxes from RPN overlapping ground truth boxes: {mean_overlapping_bboxes}')
					print(f'Training accuracy for bounding boxes: {class_acc}')
					print(f'Training precision for bounding boxes: {class_precision}')
					print(f'Training recall for bounding boxes: {class_recall}')
					print(f'Validation accuracy for bounding boxes: {val_acc}')
					print(f'Validation precision for bounding boxes: {val_precision}')
					print(f'Validation recall for bounding boxes: {val_recall}')
					print(f'Validation f1 for bounding boxes: {val_f1}')
					print(f'Elapsed time: {time.time() - start_time}')

				curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
				iter_num = 0
				start_time = time.time()

				# Use best val acc for saving model
				if curr_loss < best_loss:
					best_loss= curr_loss
					model_all.save_weights(C.model_path)

				break

		# except Exception as e:
		# 	model_all.save_weights("model_weights/emergency_model_frcnn.hdf5")
		# 	break

print('Training complete, exiting.')
