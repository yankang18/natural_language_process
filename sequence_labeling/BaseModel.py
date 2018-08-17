
import os
import tensorflow as tf

class BaseModel(object):

	def __init__(self):
		self.sess = None
		self.saver = None
		self.stop_training = False

	def initialize_session(self):
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()

	def close_session(self):
		self.sess.close()


	def load_model(self, dir_model):
		self.saver.restore(self.sess, dir_model)


	def save_model(self, dir_model):
		if not os.path.exists(dir_model):
			os.makedirs(dir_model)
		self.saver.save(self.sess, dir_model)

	def is_stop_training(self):
		return self.stop_training




	







