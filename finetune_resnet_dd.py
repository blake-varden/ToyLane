import argparse
from resnet50 import ResNet50
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, TensorBoard
import pickle
from keras.models import Model
from keras import backend as K
import keras
import numpy as np
import time
from model_history import ModelHistory
import random
from simplelmdb import Simplelmdb
import itertools as it
import numpy as np
import json


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--train", required=True,help="lmdb folder containing train data")
	parser.add_argument("--test", required=True, help="lmdb folder containing test data")
	parser.add_argument("--num_train", type=int, help="recoring location")
	parser.add_argument("--num_test", type=int, help="recoring location")
	parser.add_argument("--num_epochs",default=7, type=int, help="recoring location")
	parser.add_argument("--history", help="folder containing model history.")
	args = parser.parse_args()

class BatchHistory(Callback):
	'''Callback that records events
	into a `History` object.
	This callback is automatically applied to
	every Keras model. The `History` object
	gets returned by the `fit` method of models.
	'''
	def __init__(self, save_file, save_every_batch):
		self.save_file = save_file
		self.save_every_batch = save_every_batch

	def on_train_begin(self, logs={}):
	    self.batch = []
	    self.history = {}
	    self.epoch_history = {}
	    self.epoch = 0
	    self.batches_per_epoch = 0

	def on_epoch_end(self, epoch, logs={}):
		self.epoch +=1
		if not self.batches_per_epoch:
			self.batches_per_epoch = len(self.batch)

		for k, v in logs.items():
			# use prev value and end with new value
			history_values = [v]* (self.batches_per_epoch)
			
			self.epoch_history.setdefault(k, [0]*self.batches_per_epoch).extend(history_values)

		lr = K.get_value(self.model.optimizer.lr)
		self.epoch_history.setdefault('learning_rate', []).extend([lr] *self.batches_per_epoch)
		if self.epoch > 0:
			self.save()

	def on_batch_end(self, batch, logs={}):
		# increase the epoch on batch 0
		self.batch.append((len(self.batch), self.epoch))
		for k, v in logs.items():
			self.history.setdefault(k, []).append(v)
			lr = K.get_value(self.model.optimizer.lr)
		self.history.setdefault('learning_rate', []).append(lr)
		if (len(self.batch) > 0 and len(self.batch) % self.save_every_batch == 0):
			self.save()

	def save(self):
		data = {
			'batch' : self.batch,
			'history' : self.history,
			'epoch_history': self.epoch_history,
			'params' : self.params
		}
		with open(self.save_file, 'w') as f:
			pickle.dump(data, f)


def create_accuracy_metric(i, threshold):
	def accuracy_metric(y, pred):
		diff = y[:,i]-pred[:,i]
		abs_diff = K.abs(diff)
		passes = K.lesser(abs_diff, threshold)
		passes = K.cast(passes, 'float32')
		accuracy = K.mean(passes)
		return accuracy

	accuracy_metric.__name__ = 'accuracy_metric_' + str(i)
	return accuracy_metric

def create_model(params):
	input_tensor = Input(shape=(224, 224, 3))
	base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)
	fc_init = params['fc_init']
	x = base_model.output
	x = Flatten()(x)
	x = Dense(6, init=fc_init, name='fc5')(x)

	model = Model(input_tensor, x)
	for layer in model.layers[:len(model.layers)-1]:
	    layer.trainable = False
	return model

def data_generator(db, batch_size=32, num_records=None, shuffle=True):

	batch_xs = []
	batch_ys = []

	if num_records is None:
		num_records = db.get('num_records')
	# make num_records a multiple of batch size
	keys = range(num_records)
	keys = [str(key) for key in keys]
	if shuffle:
		random.shuffle(keys)
	while True:
		#for k, v in it.islice(db.items(),0, 512):
		for v in db.get_keys(keys):
			image_data = v['image_data']
			image_targets = v['image_targets']
			batch_ys.append(image_targets)
			batch_xs.append(image_data)
			if len(batch_xs) >= batch_size:
				yield (np.array(batch_xs), np.array(batch_ys))
				batch_xs = []
				batch_ys = []

def train_model(model_history, train_db, test_db, params):
	optimizer = params['optimizer']
	loss = params['loss']
	metrics = params['metrics']	
	nb_epoch = params['nb_epoch']
	batch_size = params['batch_size']

	thresholds = np.array([2,3,4,2,3,4])
	data_min = train_db.get('data_min')
	data_max = train_db.get('data_max')
	low = train_db.get('low')
	high = train_db.get('high')
	target_range = high - low
	data_range = data_max-data_min
	scaling = data_range/target_range
	scaled_thresholds = thresholds/scaling

	for i in range(len(scaled_thresholds)):
		threshold = scaled_thresholds[i]
		metrics.append(create_accuracy_metric(i, threshold))

	model = create_model(params)
	model.compile(optimizer=optimizer,
		loss=loss,
		metrics=metrics)

	num_train_records = int(train_db.get('num_records'))
	if 'num_train' in params and params['num_train'] is not None:
		num_train_records = params['num_train']

	num_test_records = int(test_db.get('num_records'))
	if 'num_test' in params and params['num_test'] is not None:
		num_test_records = params['num_test']	

	train_generator = data_generator(train_db, num_records=num_train_records, batch_size=batch_size, shuffle=True)
	test_generator = data_generator(test_db, num_records=num_test_records, batch_size=batch_size)
	

	history = BatchHistory(model_history.training_history, 100)
	checkpoint_save = model_history.model_save + '.{epoch:02d}-{val_loss:.2f}.hdf5'	
	checkpointer = ModelCheckpoint(filepath=checkpoint_save, verbose=1, save_best_only=True)
	tensorboard = TensorBoard(log_dir=model_history.tensorboard, write_graph=False, histogram_freq=1)
	model.fit_generator(train_generator, num_train_records, nb_epoch, 
		validation_data=test_generator, 
		nb_val_samples=num_test_records,
		callbacks=[history, checkpointer, tensorboard])

	model.save(model_history.model_save + '.hdf5')
	return model

def generate_predictions(model_history, model, train_db, test_db, params):
	batch_size = params['batch_size']
	num_train_records = int(train_db.get('num_records'))
	if 'num_train' in params and params['num_train'] is not None:
		num_train_records = params['num_train']

	num_test_records = int(test_db.get('num_records'))
	if 'num_test' in params and params['num_test'] is not None:
		num_test_records = params['num_test']	

	train_generator = data_generator(train_db, num_records=num_train_records, batch_size=1)
	test_generator = data_generator(test_db, num_records=num_test_records, batch_size=1)


	train_predictions = model.predict_generator(train_generator, num_train_records)
	test_predictions = model.predict_generator(test_generator, num_test_records)

	data = {'train' : train_predictions, 'test' : test_predictions}
	with open(model_history.predictions, 'w') as f:
		pickle.dump(data, f)

def main(args):
	"""
	Every time a model is run, a new line is added to the notes json, recording the model name, time, params, user notes. name of data file used.
	A folder with model name is created.  In that folder will reside checkpoints.  Final Models. repeated notes. predicted train and test results.

	"""
	# load data


	notes = raw_input("Please describe whats going on with this model: ")

	train_db = Simplelmdb(args.train)
	test_db = Simplelmdb(args.test)	

	num_train = args.num_train
	num_test = args.num_test	
	meta = {
		'notes' : notes,
		'train' : args.train,
		'test' : args.test,
		'num_train' : num_train,
		'num_test' : num_test
	}
	params = {
		'num_train' : num_train,
		'num_test' : num_test,
		'fc_init' : 'uniform',
		'optimizer' : 'adam',
		'loss' : 'mean_squared_error',
		'metrics' : ['mean_absolute_error'],
		'nb_epoch' : args.num_epochs,
		'batch_size' : 32
	}


	model_history  = ModelHistory(args.history, meta, params)
	model = train_model(model_history, train_db, test_db, params)
	generate_predictions(model_history, model, train_db, test_db, params)


if __name__ == '__main__':
	main(args)