import argparse
import pickle
import time
import json
import os
import shutils


class ModelHistory(object):
	def __init__(self, history_dir, meta, params):
		# directory containing all our models
		self.history_dir = history_dir
		self.notes_file = history_dir + '/history.txt'

		self.meta = meta
		self.params = params

		self.create_history_dir()
		self.model_number = self.get_model_number()
		self.model_name = 'model_' + str(self.model_num)
		self.model_dir = self.history_dir + '/' + self.model_name
		self.model_save = self.model_dir + '/model'
		self.notes_save = self.model_dir + '/notes.json'
		self.training_history = self.model_dir + '/training_history.json'
		self.predictions = self.model_dir + '/predictions.npy'
		self.update_model_history()
		self.create_model_dir()


	def create_history_dir(self):
		if not os.path.exists(self.history_dir):
    		os.makedirs(self.history_dir)

    def get_model_number(self):
		if not os.path.exists(self.notes_file):
    		return 0        	
    	with open(self.notes_file) as f:
    		model_num = len(f.readlines())
    		return model_num

    def update_model_history(self):
    	history = self.get_history()
    	with open(self.notes_file, 'a') as f:
    		line = json.dumps(history)
    		f.write(line + '\n')


    def get_history(self):
    	return {
    		'name' : self.model_name,
    		'meta' : self.meta,
    		'params' : self.params
    	}

    def create_model_dir(self):
		if not os.path.exists(self.model_dir):
			raise "Something went wrong, this model already exists: " + self.model_dir
		os.makedirs(self.model_dir)  
		history = self.get_history() 
		with open(self.notes_save, 'w') as f:
			line = json.dumps(history)
			f.write(line + '\n')