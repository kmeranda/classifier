import autograd.scipy.misc
import numpy as np
import random

def main():
	flag = "bigrams"
	trainwords = get_words('data/train', flag)
	devwords = get_words('data/dev', flag)
	testwords = get_words('data/test', flag)
	trainname = 'data/train'
	lines = open(trainname).read().strip().splitlines()
	vocab_size = 0	# num of unique words over all docs
	vocab = {}	# dict of unique words 
	speakers = []	# list of speakers

	#############################
	### Calculate word counts ###
	#############################
	for line in lines:	# iterate over each document
		words = line.split()		# split by white space into words
		words.append('<bias>')
		if words[0] not in speakers:		# if you haven't seen the speaker yet, append to array
			speakers.append(words[0])
		if flag == "bigrams":
			if len(words) > 1:
				first = "<s> " + words[1]
				end = words[-1] + " </s>"
				if first not in vocab:
					vocab[first] = 1
					vocab_size += 1
				if end not in vocab:
					vocab[end] = 1
					vocab_size += 1
			if len(words) > 1:
				prev = words[1]
			just_no = True
		for word in words[1:]:	# iterate over each word in a document
			if flag == "bigrams" and not just_no:
				bigram = prev + " " + word
				if bigram not in vocab:
					vocab[bigram] = 1
					vocab_size += 1
				prev = word
			if flag == "suffixes":
				if len(word) > 3:
					suffix = word[-3:]
				else:
					suffix = word
				if suffix not in vocab:
					vocab[suffix] = 1
					vocab_size += 1
			if word not in vocab:	# calculate |V|
				vocab[word] = 1
				vocab_size += 1
			just_no = False
	
	###################
	### Train Model ###
	###################
	# 2b
	print("2b.")
	l_rate = 0.01	# learning rate
	# init lambda(k, w) = 0
	lamb = {}
	for k in speakers:
		lamb[k] = {}
		for w in vocab:
			lamb[k][w] = 0
	# repeat until convergence to learn lambda values
	P_k_d = {}
	P_k_d_dev = {}
	for x in range(20):
		random.shuffle(lines)
		# for finding accuracy on dev files
		for i in devwords:
			# create array of words in dev doc
			words = i.split()
			#words.append('<bias>')
						
			P_k_d_dev[i] = logP(lamb, devwords[i], words[0])
		for i in trainwords:
			# create array of words in doc
			words = i.split()
			#words.append('<bias>')
			P_k_d[i] = logP(lamb, trainwords[i], words[0])
			for w in trainwords[i]:
				lamb[words[0]][w] += l_rate
				for k in speakers:
					lamb[k][w] += -1*l_rate*np.exp(-1*P_k_d[i][k])
		print("\niteration ", x+1, ":")
		prob_sum = 0
		for d in P_k_d:
			prob_sum += P_k_d[d][d.split()[0]]
		print("-log(P(k|d)) for train: ", prob_sum)
		print("accuracy: ", accuracy(P_k_d_dev))
	print("lambda(trump):", lamb['trump']['<bias>'])
	print("lambda(clinton):", lamb['clinton']['<bias>'])
	print("lambda(trump, country):", lamb['trump']['country'])
	print("lambda(clinton, country):", lamb['clinton']['country'])
	print("lambda(trump, president):", lamb['trump']['president'])
	print("lambda(clinton, president):", lamb['clinton']['president'])

	#2c
	print("\n2c.")
	devlines = open('data/dev').read().strip().splitlines()
	for k in P_k_d_dev[devlines[0]]:
		print(k, ": ", np.exp(-1*P_k_d_dev[devlines[0]][k]))
	
	#2d
	# for finding accuracy on test files
	P_k_d_test = {}
	for i in testwords:
		# create array of words in doc
		words = i.split()
		P_k_d_test[i] = logP(lamb, testwords[i], words[0])
	print("\n2d.")
	print("accuracy: ", accuracy(P_k_d_test))

# 2a function
def logP(model, doc, correct_k):
	# recalculate P(k|d)
	s_kd = {}
	P_k_d = {}
	for k in model:
		s_kd[k] = 0
		for w in doc:	# doc is all words in line - the speaker name
			if w in model[k]:
				s_kd[k] += model[k][w]
	# need to wait until s_kd fully calculated
	for k in model:
		P_k_d[k] = autograd.scipy.misc.logsumexp(np.array(list(s_kd.values()))) - s_kd[k]
	return P_k_d	# dict containing -log P(k|d) for all speakers for the document

def accuracy(P_k_d):	# calculate accuracy (# right/total #)
	total_num = 0
	total_correct = 0
	for d in P_k_d:
		if max_prob(P_k_d[d]) == d.split()[0]:
			total_correct +=1
		total_num += 1
	return total_correct/total_num

def max_prob(P_k_d):	# determine most likely speaker
	max_speaker = None
	max_prob = 0
	for speaker in P_k_d:
		if np.exp(-1*P_k_d[speaker]) > max_prob:	#because its -log prob
			max_speaker = speaker
			max_prob = np.exp(-1*P_k_d[speaker])
	return max_speaker

def get_words(filename, flag):
	lines = open(filename).read().strip().splitlines()
	all_words = {}
	for i in lines:
		words = i.split()
		words.append("<bias>")
		all_ngrams = words[1:]
		if flag == "bigrams":
			bigrams = []
			if len(words) > 1:
				first = "<s> " + words[1]
				end = words[-1] + " </s>"
				bigrams.append(first)
				bigrams.append(end)
				prev = words[1]
				for word in words[2:]:	# iterate over each word in a document
					bigram = prev + " " + word
					bigrams.append(bigram)
					prev = word
			all_ngrams.extend(bigrams)
		if flag == "suffixes":
			suffixes = []
			for word in words[1:]:
				if len(word)>3:
					suffix = word[-3:]
				else:
					suffix = word
				suffixes.append(suffix)
			all_ngrams.extend(suffixes)
		all_words[i] = all_ngrams
	return all_words

if __name__ == '__main__':
	main()
