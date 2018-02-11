import sys
import random
import re
import time
import torch 
from torch.autograd import Variable
import torch.nn.init as init
import gc

sub = '-'*20
class W2vCharacterTable(object):
	def __init__(self, w2v_path, ini_char = '</i>', unk_char='<unk>'):
		super(W2vCharacterTable, self).__init__()
		w2v_file = open(w2v_path,'r')
		self._dict_size = 0
		self._emb_size = 0
		self._word2index = {}
		self._index2word = {}
		self._ini_char = ini_char 
		self._unk_char = unk_char

		list_emb = []
		idx = 0
		for line in w2v_file:
			line_list = line.strip().split(' ')
			if len(line_list) == 2:
				self._dict_size = int(line_list[0])
				self._emb_size = int(line_list[1])
			else:
				self._word2index[line_list[0]] = idx
				self._index2word[idx] = line_list[0]
				tmp_emb = [float(item) for item in line_list[1:]]
				list_emb.append(torch.FloatTensor(tmp_emb).unsqueeze(0))
				idx += 1
		w2v_file.close()

		if self._unk_char not in self._word2index:
			self._word2index[self._unk_char] = self._dict_size
			self._index2word[self._dict_size] = self._unk_char
			self._dict_size += 1
			list_emb.append(init.uniform(torch.FloatTensor(1,self._emb_size),-0.1,0.1))

		if self._ini_char not in self._word2index:
			self._word2index[self._ini_char] = self._dict_size
			self._index2word[self._dict_size] = self._ini_char
			self._dict_size += 1
			list_emb.append(init.uniform(torch.FloatTensor(1,self._emb_size),-0.1,0.1))

		self._emb_matrix = torch.cat(tuple(list_emb),0).cuda()
		
	def getDictSize(self):
		return self._dict_size

	def getEmbSize(self):
		return self._emb_size

	def getEmbMatrix(self):
		return self._emb_matrix

	def getWord2Index(self):
		return self._word2index

	def getIndex2Word(self):
		return self._index2word

	def getIniChar(self):
		return self._ini_char

	def getUnkChar(self):
		return self._unk_char

def paddingSenten(sentence,max_senten_len):
	senten_list = sentence.split('\t')
	if len(senten_list) <= max_senten_len:
		senten_list.extend(["</s>"]*(max_senten_len-len(senten_list)))
	else:
		senten_list = senten_list[0:max_senten_len]
	return '\t'.join(senten_list)

def paddingPair(corpus_pair,max_senten_len,max_context_size):
	padded_pair = []
	context = corpus_pair[0]
	reply_sentence = corpus_pair[1]
	pad_list = [0]*len(context)
	if len(context) <= max_context_size:
		pad_list = [1]*(max_context_size-len(context)) + pad_list
		context = ['<unk>']*(max_context_size-len(context)) + context
	else:
		pad_list = pad_list[-max_context_size:]
		context = context[-max_context_size:]
	context = [paddingSenten(senten,max_senten_len) for senten in context]
	reply_sentence = paddingSenten(reply_sentence,max_senten_len)

	padded_pair.append(context)
	padded_pair.append(reply_sentence)
	padded_pair.append(pad_list)
	return padded_pair

def paddingPairs(corpus_pairs,max_senten_len,max_context_size):
	padded_pairs = [paddingPair(pair,max_senten_len,max_context_size) for pair in corpus_pairs] 
	return padded_pairs

def readingCorpus(train_file):
	train_file = open(train_file,'r')
	list_pairs = []
	tmp_pair = []
	for line in train_file:
		line = line.strip('\n')
		if line == sub:
			list_pairs.append(tmp_pair)
			tmp_pair = []
		else:
			tmp_pair.append(line)
	train_file.close()

	corpus_pairs = []
	corpus_pair = []
	max_con_size = 0
	min_con_size = 100
	for pair in list_pairs:
		if len(pair) >= 3:
			corpus_pair.append(pair[0:-1]) # context
			corpus_pair.append(pair[-1]) # post
			corpus_pairs.append(corpus_pair)
			corpus_pair = []
			max_con_size = max(max_con_size,len(pair[0:-1]))
			min_con_size = min(min_con_size,len(pair[0:-1]))
		else:
			pass
	return corpus_pairs

def filteringSenten(word2index,sentence,unk_char,ini_char):
	senten_list = sentence.split('\t')
	filtered_senten_list = []
	for word in senten_list:
		if word in word2index:
			filtered_senten_list.append(word)
		else:
			filtered_senten_list.append(unk_char)
	if filtered_senten_list == []:
		filtered_senten_list = [ini_char]
	return '\t'.join(filtered_senten_list)

def filteringPair(word2index,corpus_pair,unk_char,ini_char):
	filtered_pair = []
	context = corpus_pair[0]
	reply_sentence = corpus_pair[1]
	context = [filteringSenten(word2index,senten,unk_char,ini_char) for senten in context]
	reply_sentence = filteringSenten(word2index,reply_sentence,unk_char,ini_char)
	filtered_pair.append(context)
	filtered_pair.append(reply_sentence)
	return filtered_pair

def filteringPairs(word2index,corpus_pairs,unk_char,ini_char):
	filtered_pairs = [filteringPair(word2index,pair,unk_char,ini_char) for pair in corpus_pairs]
	return filtered_pairs

def readingData(ctable, train_file, max_senten_len, max_context_size):
	print ("loading corpus...")
	corpus_pairs = readingCorpus(train_file)
	print ("preprocessing...")
	corpus_pairs = filteringPairs(ctable.getWord2Index(),corpus_pairs,ctable.getUnkChar(),ctable.getIniChar())
	corpus_pairs = paddingPairs(corpus_pairs,max_senten_len,max_context_size)

	print ("num of pair:",len(corpus_pairs))
	return ctable,corpus_pairs

def buildingPairsBatch(corpus_pairs,batch_size,shuffle=True):
	if shuffle:
		print ("shuffle...")
		random.shuffle(corpus_pairs)
	pairs_batches = []
	batch = []
	for idx,pair in enumerate(corpus_pairs):
		batch.append(pair)
		if (idx+1)%batch_size == 0 or (idx+1)==len(corpus_pairs):
			pairs_batches.append(batch)
			batch = []
	num_batches = len(pairs_batches)

	last_batch = pairs_batches[-1]
	if len(last_batch) < batch_size:
		for i in range(len(last_batch),batch_size):
			pairs_batches[-1].append(last_batch[-1])

	return pairs_batches,num_batches

def indexesFromSentence(word2index, sentence):
	indexes = []
	for word in sentence.split('\t'):
		try:
			index = word2index[word]
			indexes.append(index)
		except Exception:
			print ("sentence:",sentence)
			if word == '':
				print ("Error!")
			exc_type, exc_value, exc_traceback = sys.exc_info()
			print (exc_type,"[",exc_traceback.tb_lineno,"]",str(exc_value))
	return indexes

def tensorFromSentence(word2index,sentence):
	indexes = indexesFromSentence(word2index,sentence)
	tensor = torch.LongTensor(indexes).view(-1,1)
	return tensor

def tensorsFromContext(word2index,context):
	list_context_tensor = [tensorFromSentence(word2index,senten) for senten in context]
	return list_context_tensor

def tensorsFromPair(word2index,pair):
	list_context_tensor = [tensorFromSentence(word2index,senten) for senten in pair[0]]
	reply_tensor = tensorFromSentence(word2index,pair[1])
	pad_matrix = torch.ByteTensor(pair[2])
	return list_context_tensor,reply_tensor,pad_matrix

def tensorsFromContextPair(word2index,pair):
	list_context_tensor = [tensorFromSentence(word2index,senten) for senten in pair[0]]
	pad_matrix = torch.ByteTensor(pair[1])
	return list_context_tensor,pad_matrix

def getTensorsPairsBatch(word2index,pairs_batches,max_context_size):
	for batch in pairs_batches:
		list_reply = []
		list_context = [[] for i in range(max_context_size)]
		pad_matrixs = []

		for pair in batch:
			list_context_tensor,reply_tensor,pad_matrix = tensorsFromPair(word2index,pair)
			list_reply.append(reply_tensor)
			for idx,context_tensor in enumerate(list_context_tensor):
				list_context[idx].append(context_tensor.t())
			pad_matrixs.append(pad_matrix.unsqueeze(0))

		contexts_tensor_batch = [torch.cat(tuple(context),0) for context in list_context]

		pad_matrix_batch = torch.cat(tuple(pad_matrixs),0)
		reply_tensor_batch = torch.cat(tuple(list_reply),1)
		
		yield reply_tensor_batch, contexts_tensor_batch, pad_matrix_batch
		del reply_tensor_batch
		del contexts_tensor_batch
		del pad_matrix_batch
		gc.collect()

def getTensorsContextPairsBatch(word2index,pairs_batches,max_context_size):
	for batch in pairs_batches:
		list_context = [[] for i in range(max_context_size)]
		pad_matrixs = []

		for pair in batch:
			list_context_tensor,pad_matrix = tensorsFromContextPair(word2index,pair)
			for idx,context_tensor in enumerate(list_context_tensor):
				list_context[idx].append(context_tensor.t())
			pad_matrixs.append(pad_matrix.unsqueeze(0))

		contexts_tensor_batch = [torch.cat(tuple(context),0) for context in list_context]

		pad_matrix_batch = torch.cat(tuple(pad_matrixs),0)
		
		yield contexts_tensor_batch, pad_matrix_batch
		del contexts_tensor_batch
		del pad_matrix_batch
		gc.collect()