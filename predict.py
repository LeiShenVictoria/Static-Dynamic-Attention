import sys
import os
import random
import re
import time
import torch 
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
from static_model import StaticModel
from dyna_model import DynamicModel
from data_utils import *

sub = '-'*20
def init_command_line(argv):
	from argparse import ArgumentParser
	usage = "train"
	description = ArgumentParser(usage)
	description.add_argument("--test_data", type=int, default=1)
	
	description.add_argument("--max_context_size", type=int, default=9)
	description.add_argument("--batch_size", type=int, default=80)
	description.add_argument("--hidden_size", type=int, default=512)
	description.add_argument("--max_senten_len", type=int, default=15)
	description.add_argument("--type_model", type=int, default=1)

	description.add_argument("--dropout", type=float, default=0.5)

	description.add_argument("--teach_forcing", type=int, default=1)
	description.add_argument("--weights", type=str, default=None)
	return description.parse_args(argv)

opts = init_command_line(sys.argv[1:])
print ("Configure:")
if opts.test_data:
	print (" Opensubtitle dataset for testing")
	test_file = "./data/open_test_sessions.txt"
	w2v_file = "./data/open_train_all_100e.w2v"
else:
	print (" Ubuntu dataset for testing")
	test_file = "./data/ubuntu_test_sessions.txt"
	w2v_file = "./data/ubuntu_train_all_100e.w2v"
print (" test_file:",test_file)
print (" w2v_file:",w2v_file)

print (" max_context_size:",opts.max_context_size)
print (" batch_size:",opts.batch_size)
print (" hidden_size:",opts.hidden_size)
print (" max_senten_len:",opts.max_senten_len)
if opts.type_model:
	print (" static model")
else:
	print (" dynamic model")

print (" dropout:",opts.dropout)

print (" teach_forcing:",opts.teach_forcing)
print (" weights:",opts.weights)
print ("")

def readingTestCorpus(test_file_path):
	print ("reading...")
	test_file = open(test_file_path,'r')
	list_pairs = []
	tmp_pair = []
	for line in test_file:
		line = line.strip('\n')
		if line == sub:
			list_pairs.append(tmp_pair)
			tmp_pair = []
		else:
			tmp_pair.append(line)
	test_file.close()

	test_contexts = []
	test_replys = []
	max_con_size =  0
	min_con_size = 10000
	for pair in list_pairs:
		if len(pair) >= 3:
			test_contexts.append(pair[0:-1])
			test_replys.append(pair[-1])
			max_con_size = max(len(pair[0:-1]),max_con_size)
			min_con_size = min(len(pair[0:-1]),min_con_size)
		else:
			pass
	print (max_con_size)
	print (min_con_size)
	return test_contexts,test_replys

def preProcess(word2index,test_contexts,unk_char,ini_char,max_senten_len,max_context_size):
	print ("preprocessing...")
	filter_test_contexts = []
	for context in test_contexts:
		filter_context = [filteringSenten(word2index,senten,unk_char,ini_char) for senten in context]
		filter_test_contexts.append(filter_context)

	padded_test_pairs = []
	for context in filter_test_contexts:
		pad_list = [0]*len(context)
		if len(context) <= max_context_size:
			pad_list = [1]*(max_context_size-len(context)) + pad_list
			context = ['<unk>']*(max_context_size-len(context)) + context
		else:
			pad_list = pad_list[-max_context_size:]
			context = context[-max_context_size:]
		padded_context = [paddingSenten(senten,max_senten_len) for senten in context]
		padded_test_pairs.append([padded_context,pad_list])

	return padded_test_pairs


def predictSentences(index2word,unk_char,ini_char,ini_idx,model,test_pairs,batch_size,max_senten_len,max_context_size):
	model.eval()
	pairs_batches,num_batches = buildingPairsBatch(test_pairs,batch_size,shuffle=False)
	print ("")
	print ("num of batch:",num_batches)
	
	predict_sentences = []
	idx_batch = 0
	for contexts_tensor_batch, pad_matrix_batch in getTensorsContextPairsBatch(word2index,pairs_batches,max_context_size):
		predict_batch = model.predict(contexts_tensor_batch,index2word,pad_matrix_batch,ini_idx,sep_char='\t')
		predict_sentences.extend(predict_batch)
		idx_batch += 1

	predict_sentences = predict_sentences[0:len(test_pairs)]
	return predict_sentences

if __name__ == '__main__':
	ini_char = '</i>'
	unk_char = '<unk>'
	t0 = time.time()
	print ("loading word2vec...")
	ctable = W2vCharacterTable(w2v_file,ini_char,unk_char)
	print(" dict size:",ctable.getDictSize())
	print (" emb size:",ctable.getEmbSize())
	print (time.time()-t0)
	print ("")

	if opts.type_model:
		model = StaticModel(ctable.getDictSize(),ctable.getEmbSize(),opts.hidden_size,opts.batch_size,opts.dropout,
						opts.max_senten_len,opts.teach_forcing).cuda()
	else:
		model = DynamicModel(ctable.getDictSize(),ctable.getEmbSize(),opts.hidden_size,opts.batch_size,opts.dropout,
						opts.max_senten_len,opts.teach_forcing).cuda()

	if opts.weights != None:
		print ("load model parameters...")
		model.load_state_dict(torch.load(opts.weights))
	else:
		print ("No model parameters!")
		exit()

	test_contexts,test_replys = readingTestCorpus(test_file)
	print ("len(test_contexts):",len(test_contexts))
	print ("len(test_replys):",len(test_replys))

	word2index = ctable.getWord2Index()
	test_pairs = preProcess(word2index,test_contexts,unk_char,ini_char,opts.max_senten_len,opts.max_context_size)
	print ("len(test_pairs):",len(test_pairs))
	
	print ("start predicting...")
	ini_idx = word2index[ini_char]
	predict_sentences = predictSentences(ctable.getIndex2Word(),unk_char,ini_char,ini_idx,model,test_pairs,
										opts.batch_size,opts.max_senten_len,opts.max_context_size)

	print ("writing...")
	if not os.path.exists('./result/'):
		os.mkdir('./result/')
	pred_res_file = open("./result/open_pred_result",'w')
	pred_ans_file = open("./result/open_pred_answer",'w')
	for idx,senten in enumerate(predict_sentences):
		test_context = test_contexts[idx]
		for test_post in test_context:
			pred_res_file.write(test_post+'\n')
		pred_res_file.write(senten+'\n')
		pred_res_file.write(sub+'\n')
		senten_l = [c for c in senten.split('\t') if c != '</s>']
		pred_ans_file.write(' '.join(senten_l)+' </s>'+'\n')

	pred_res_file.close()
	pred_ans_file.close()
	print ("end")

