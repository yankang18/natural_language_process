

def minibatch(dataset, batch_size):
	xbatch, ybatch = [], []
	for word, tag in dataset:
		if len(xbatch) == batch_size:
			yield xbatch, ybatch
			xbatch, ybatch = [], []
			
		xbatch += [word]
		ybatch += [tag]
	
	if len(xbatch) != 0:
		yield xbatch, ybatch  


def pad_sequences(sequences):
	max_len = max(map(lambda x:len(x), sequences))
	
	sequences_pad, sequences_length = [], []
	for seq in sequences:
		seq = list(seq)
		seq_ = seq[:max_len] + [0]*max(max_len - len(seq), 0)
		sequences_pad += [seq_]
		sequences_length += [min(len(seq), max_len)]
	return sequences_pad, sequences_length