import numpy as np

# We create a simple class, called EarlyStoppingCheckPoint, that combines simplified version of EarlyStoping and ModelCheckPoint classes from Keras.
# References:
# https://keras.io/callbacks/
# https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L458
# https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L358

class EarlyStoppingCheckPoint(object):

	def __init__(self, file_path, monitor, patience):

		self.filepath = file_path
		self.monitor = monitor
		self.patience = patience

		self.wait = 0
		self.stopped_epoch = 0

	def set_model(self, model):
		self.model = model


	def on_train_begin(self):
		self.wait = 0
		self.stopped_epoch = 0
		self.best = -np.Inf

	def on_epoch_end(self, epoch, logs=None):

		current = logs.get(self.monitor)
		if current is None:
			print('monitor does not available in logs')
			return
		
		if current > self.best:
			self.best = current
			self.model.save_model(self.filepath)
			self.wait = 0
		else:
			self.wait += 1
			if self.wait >= self.patience:
				self.stopped_epoch = epoch
				self.model.stop_training = True






