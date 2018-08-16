import tensorflow as tf
import numpy as np


class BiLSTM_CRF(BaseModel):

	def __init__(self):

	def build(self, learning_rate, dropout_rate):

		self.learning_rate = learning_rate
		self.dropout_rate = dropout_rate

		self.pretrained_embedding = False
		self.use_crf = True

		self.add_placeholders()
		self.add_wordembedding()
		self.add_bilstm_layer()
		self.add_decoder()
		self.add_loss()
		self.add_projection_op()
		self.add_train_op()
		self.initialize_session()


	def add_placeholders(self):
		self.word_ids = tf.placeholder(shape=[None, None], dtype=tf.int32, name='word_ids')
		self.sequence_lengths = tf.placeholder(shape=[None], dtype=tf.int32, name='sequence_lengths')
		self.labels = tf.placeholder(shape=[None, None], dtype=tf.int32, name='label')

		self.learning_rate = tf.placeholder( dtype=tf.float32, name='learning_rate')
		self.dropout_rate = tf.placeholder( dtype=tf.float32, name='dropout_rate')

	def add_wordembedding(self):

		if not self.pretrained_embedding:
    		word_embedding_matrix = tf.get_variable(shape=[word_num, dim_word], 
                                        dtype=tf.float32, 
                                        name='word_embedding_matrix')
		else:
    		filename_glove = "data/glove.6B/glove.6B.{}d.txt".format(dim_word)
    		embedding_matrix = np.zeros((word_num, dim_word))
    		with open(filename_glove) as f:
        		for line in f:
            		line = line.strip().split(' ')
            		word = line[0]
            		if word in word2index:
                		embedding_matrix[word2index[word]] = np.asarray(line[1:])
    
    		word_embedding_matrix = tf.Variable(embedding_matrix,
                                        dtype=tf.float32, 
                                        name='pretrained_word_embedding_matrix',
                                        trainable=True)

		self.word_embedding = tf.nn.embedding_lookup(word_embedding_matrix, 
                                        self.word_ids, 
                                        name='word_embedding_lookup')


	def add_bilstm_layer(self):

		with tf.variable_scope('bi_lstm_encoder'):
    		cell_fw = tf.contrib.rnn.LSTMCell(hidden_size_lstm)
    		cell_bw = tf.contrib.rnn.LSTMCell(hidden_size_lstm)
    
    		(output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                                		cell_bw, 
                                                                		self.word_embedding, 
                                                                		sequence_length=self.sequence_lengths, 
                                                                		dtype=tf.float32)
    		output = tf.concat([output_fw, output_bw], axis=-1)
    		self.output = tf.nn.dropout(output, self.dropout_rate)



    def add_decoder(self):

		with tf.variable_scope('decoder'):
	    	W = tf.get_variable(shape=[2*hidden_size_lstm, tag_num], dtype=tf.float32, name='proj_W')
	    	b = tf.get_variable(shape=[tag_num], dtype=tf.float32, name='proj_b')
	    
	    	time_steps = tf.shape(output)[1]
	    	output = tf.reshape(self.output, [-1, 2*hidden_size_lstm])
	    	pred = tf.matmul(output, W) + b
	    	self.logits = tf.reshape(pred, [-1, time_steps, tag_num])
	    	# self.logits_shape = tf.shape(logits)


	 def add_loss(self):

	 	with tf.variable_scope('loss'):
    		if self.use_crf:
        		log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(self.logits, 
                                                                         		 self.labels, 
                                                                         		 self.sequence_lengths) 
        		self.loss = tf.reduce_mean(-log_likelihood)
        		# log_likelihood_shape = tf.shape(log_likelihood)
        		# trans_params_shape = tf.shape(trans_params)
    		else:
        		losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
        		mask = tf.sequence_mask(self.sequence_lengths)
        		losses = tf.boolean_mask(losses, mask)
        		self.loss = tf.reduce_mean(losses)


	def add_projection_op(self):
    	with tf.variable_scope('projection'):
    		if not use_crf:
        		self.label_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)


    def add_train_op(self):
    	with tf.variable_scope("train_step"):
    		optimizer = tf.train.AdamOptimizer(self.learning_rate)
			self.train_op = optimizer.minimize(loss)


	# def add_init_op(self):
	# 	self.init_op = tf.global_variables_initializer()


	def fit(self, train_ds, epochs, batch_size, earlyStoppingCheckPoint):
		"""

		"""

		self.initialize_session()

		earlyStoppingCheckPoint.set_model(self)
		earlyStoppingCheckPoint.on_train_begin()
    	for ep in range(epochs):
        	metrics = self.run_one_epoch(ep, train_ds, valid_ds, batch_size)
        	earlyStoppingCheckPoint.on_epoch_end(ep, metrics)

        	if self.stop_training == True:
        		break

    	self.close_session()


    def run_one_epoch(ep, train_ds, valid_ds, batch_size):

    	losses = []
        i = 0
        score = 0
        for xbatch, ybatch in minibatch(train_ds, batch_size):
        	i+=1
            word_seq, sequence_len = pad_sequences(xbatch)
            target_seq, _ = pad_sequences(ybatch)

            # build feed dictionary
            feed = {
                		word_ids: word_seq,
                		labels: target_seq,
                		sequence_lengths: sequence_len,
                		learning_rate:lr,
                		dropout_rate:dr
            		}
            
            _, train_loss = self.sess.run([self.train_op, self.loss], feed_dict=feed)
            losses += [train_loss]
            
            if i % 10 ==0:
            	print('ep:', ep, 'iter:', i, 'loss:', np.mean(losses))
            if i % 50 ==0:
                score, _ = self.run_validation(valid_ds, sess, batch_size)
                print('accuracy', score)

        if score == 0:
        	score, _ = self.run_validation(valid_ds, sess, batch_size)

        metrics = {}
        metrics['acc'] = score
        return metrics


    def run_validation(self, valid_dataset, sess, batch_size):
	    accs = []
	    ret = []
	    for xbatch, labels in minibatch(valid_dataset, batch_size):
	    	word_seq, sequence_len = pad_sequences(xbatch)
	        
	        feed = {
	                word_ids: word_seq,
	                sequence_lengths: sequence_len,
	                dropout_rate:1.0
	               }
	        
	        if self.use_crf:
	            viterbi_sequences = []
	            logits_v, trans_params_v = sess.run([logits, trans_params], feed_dict=feed)
	            for logit, seq_length in zip(logits_v, sequence_len):
	                logit = logit[:seq_length]
	                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params_v)
	                viterbi_sequences += [viterbi_seq]
	            labels_pred_v = viterbi_sequences
	        else:
	            labels_pred_v = sess.run(label_pred, feed_dict=feed)
	            
	        for words, lab, lab_pred, seq_length in zip(xbatch, labels, labels_pred_v, sequence_len):
	            lab = lab[:seq_length]
	            lab_pred = lab_pred[:seq_length]
	            acc = [a==b for (a, b) in zip(lab, lab_pred)]
	            ret.append((words, lab, lab_pred, acc))
	            accs+=acc

	    overall_acc = np.mean(accs)
	    
	    return overall_acc, ret






