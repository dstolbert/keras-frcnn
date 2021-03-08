import re
import sys
import time
import pickle
import random
import pprint
import numpy as np
from optparse import OptionParser
from sklearn.model_selection import train_test_split

# TF
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Progbar
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

# Internal
from keras_frcnn import config, data_generators
from keras_frcnn import losses as loss_funcs
import keras_frcnn.roi_helpers as roi_helpers

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

parser = OptionParser()

parser.add_option("-p", "--path", dest="train_path", help="Path to training data.")
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc",
				default="simple")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois", help="Number of RoIs to process at once.", default=32)
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='vgg')
parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--rot", "--rot_90", dest="rot_90", help="Augment with 90 degree rotations in training. (Default=false).",
				  action="store_true", default=False)
parser.add_option("--num_epochs", type="int", dest="num_epochs", help="Number of epochs.", default=2000)
parser.add_option("--config_filename", dest="config_filename", help=
				"Location to store all the metadata related to the training (to be used when testing).",
				default="config.pickle")
parser.add_option("--output_weight_path", dest="output_weight_path", help="Output path for weights.", default='./model_weights/model_frcnn.hdf5')
parser.add_option("--input_weight_path", dest="input_weight_path", help="Input path for weights. If not specified, will try to load default weights provided by keras.")

(options, args) = parser.parse_args()

# pass the settings from the command line, and persist them in the config object
C = config.Config()

if not options.train_path:   # if filename is not given
	options.train_path = C.data_dir

if options.parser == 'pascal_voc':
	from keras_frcnn.pascal_voc_parser import get_data
elif options.parser == 'simple':
	from keras_frcnn.simple_parser import get_data
else:
	raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")



C.use_horizontal_flips = bool(options.horizontal_flips)
C.use_vertical_flips = bool(options.vertical_flips)
C.rot_90 = bool(options.rot_90)

C.model_path = options.output_weight_path
model_path_regex = re.match("^(.+)(\.hdf5)$", C.model_path)
if model_path_regex.group(2) != '.hdf5':
	print('Output weights must have .hdf5 filetype')
	exit(1)
C.num_rois = int(options.num_rois)

if options.network == 'vgg':
	C.network = 'vgg'
	C.base_net_weights = C.VGG_BASE
	from keras_frcnn import vgg as nn
elif options.network == 'resnet50':
	from keras_frcnn import resnet as nn
	C.network = 'resnet50'
	C.base_net_weights = C.RESNET_BASE
else:
	print('Not a valid model')
	raise ValueError

all_imgs, classes_count, class_mapping = get_data(options.train_path)
train_imgs, val_imgs = train_test_split(all_imgs, test_size=0.2, random_state=C.seed)

if 'bg' not in classes_count:
	classes_count['bg'] = 0
	class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

inv_map = {v: k for k, v in class_mapping.items()}

print('Training images per class:')
pprint.pprint(classes_count)
print(f'Num classes (including bg) = {len(classes_count)}')

config_output_filename = options.config_filename

with open(config_output_filename, 'wb') as config_f:
	pickle.dump(C,config_f)
	print(f'Config has been written to {config_output_filename}, and can be loaded when testing to ensure correct results')

random.shuffle(train_imgs)

num_imgs = len(train_imgs)

print(f'Num train samples {len(train_imgs)}')
print(f'Num val samples {len(val_imgs)}')

data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, "tf", mode='train')
data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length,"tf", mode='val')

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

rpn_optimizer = Adam(lr=1e-5)
class_optimizer = Adam(lr=1e-5)

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

vis = True

# METRICS
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

def f1_metric(y_pred, y_true):
    # y_true = y_true[:, :, 1:]
    # y_pred = y_pred[:, :, 1:]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

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
	X2, Y1, Y2, _ = roi_helpers.calc_iou(R, img_data, C, class_mapping)

	if X2 is None:
		rpn_accuracy_rpn_monitor.append(0)
		rpn_accuracy_for_epoch.append(0)
		return loss_rpn, []

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

	loss_class = [loss_class_0, loss_class_1, train_acc_metric.result()]
	grads = tape.gradient([loss_class_0, loss_class_1], model_classifier.trainable_weights)
	class_optimizer.apply_gradients(zip(grads, model_classifier.trainable_weights))

	return loss_rpn, loss_class


# Get validation accuracy after each epoch
def computeValidation():

	f1s = np.zeros((len(val_imgs)))
	for i,_ in enumerate(val_imgs):

		X, Y, img_data = next(data_gen_val)
		
		# RPN
		P_rpn = model_rpn(X)

		R = roi_helpers.rpn_to_roi(P_rpn[0].numpy(), P_rpn[1].numpy(), C, "tf", use_regr=True, overlap_thresh=0.7, max_boxes=300)
		# note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
		X2, Y1, _, _ = roi_helpers.calc_iou(R, img_data, C, class_mapping)

		if X2 is None:
			rpn_accuracy_rpn_monitor.append(0)
			rpn_accuracy_for_epoch.append(0)
			return loss_rpn, []

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

		# Classification model
		P_class = model_classifier([X, X2[:, sel_samples, :]])
		
		# Update classification accuracy metric
		val_acc_metric.update_state(P_class[0], Y1[:, sel_samples, :])
		f1s[i] = f1_metric(P_class[0], Y1[:, sel_samples, :])

	# Get result and reset metric
	res = val_acc_metric.result()
	val_acc_metric.reset_states()

	return res, np.mean(f1s)

best_loss = 100000000
# TRAINING LOOP
for epoch_num in range(num_epochs):

	progbar = Progbar(epoch_length)
	print(f'\nEpoch {epoch_num + 1}/{num_epochs}')

	while True:

		try:

			if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
				mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
				rpn_accuracy_rpn_monitor = []
				print(f'Average number of overlapping bounding boxes from RPN = {mean_overlapping_bboxes} for {epoch_length} previous iterations')
				if mean_overlapping_bboxes == 0:
					print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')


			X, Y, img_data = next(data_gen_train)
			loss_rpn, loss_class = train_step(X, Y, img_data)

			if len(loss_class) == 0:
				continue

			losses[iter_num, 0] = loss_rpn[0]
			losses[iter_num, 1] = loss_rpn[1]

			losses[iter_num, 2] = loss_class[0]
			losses[iter_num, 3] = loss_class[1]
			losses[iter_num, 4] = loss_class[2]

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
				class_acc = np.mean(losses[:, 4])
				val_acc, val_f1 = computeValidation()

				mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
				rpn_accuracy_for_epoch = []

				if C.verbose:
					print(f'\nMean number of bounding boxes from RPN overlapping ground truth boxes: {mean_overlapping_bboxes}')
					print(f'Training accuracy for bounding boxes from RPN: {class_acc}')
					print(f'Validation accuracy for bounding boxes from RPN: {val_acc}')
					print(f'Validation F1 for bounding boxes from RPN: {val_f1}')
					print(f'Loss RPN classifier: {loss_rpn_cls}')
					print(f'Loss RPN regression: {loss_rpn_regr}')
					print(f'Loss Detector classifier: {loss_class_cls}')
					print(f'Loss Detector regression: {loss_class_regr}')
					print(f'Elapsed time: {time.time() - start_time}')

				curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
				iter_num = 0
				start_time = time.time()

				# Use best val acc for saving model
				if curr_loss < best_loss:
					best_loss= curr_loss
					model_all.save_weights(C.model_path)

				break
		except:
			model_all.save_weights("model_weights/emergency_model_frcnn.hdf5")
			break

print('Training complete, exiting.')
