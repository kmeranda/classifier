import autograd.scipy.misc
import numpy as np

def main():
	flag = "none"
	filename = 'data/train'	# read in training data
	lines = open(filename).read().strip().splitlines()
	all_words = 0	# num of total words over all docs
	all_docs = 0	# num of total documents
	vocab_size = 0	# num of unique words over all docs
	vocab = {}	# dict of unique words 
	speakers = {}	# dict of speakers with dict of counts of words they say
	speaker_word_count = {}	# dict of sum(c(k,w))
	speaker_doc_count = {} # dict of sum(c(k,d))

	#############################
	### Calculate word counts ###
	#############################
	for line in lines:	# iterate over each document
		words = line.split()		# split by white space into wordsa
		if words[0] not in speakers:		# if you haven't seen the speaker yet, create a new dict to store word counts
			speakers[words[0]] = {}
			speaker_word_count[words[0]] = 0
			speaker_doc_count[words[0]] = 0
		speaker_doc_count[words[0]] += 1
		all_docs += 1
		if flag == "bigrams":
			if len(words) > 1:	# at least one word in doc, add start and end bigrams
				first = "<s> " + words[1]
				end = words[-1] + " </s>"
				all_words += 2
				speaker_word_count[words[0]] += 2
				if first not in vocab:	# calculate |V|
					vocab[first] = 1
					vocab_size += 1
				if end not in vocab:	# calculate |V|
					vocab[end] = 1
					vocab_size += 1
				if first in speakers[words[0]]:	# if you haven't seen the word before, initialize word count
					speakers[words[0]][first] += 1
				else:
					speakers[words[0]][first] = 1
				if end in speakers[words[0]]:	# if you haven't seen the word before, initialize word count
					speakers[words[0]][end] += 1
				else:
					speakers[words[0]][end] = 1
		if len(words)>1:
			prev = words[1]
		just_no = True
		for word in words[1:]:	# iterate over each word in a document
			if flag == "suffixes":
				if len(word) > 3:	# length of word enough to split off last 3 letters
					suffix = word[-3:]
				else:	# suffix bigger than word, so whole word is suffix
					suffix = word
				all_words += 1
				speaker_word_count[words[0]] += 1
				if suffix not in vocab:	# calculate |V|
					vocab[suffix] = 1
					vocab_size += 1
				if suffix in speakers[words[0]]:	# if you haven't seen the word before, initialize word count
					speakers[words[0]][suffix] += 1
				else:
					speakers[words[0]][suffix] = 1
				
			if flag == "bigrams" and not just_no:
				bigram = prev + " " + word
				all_words += 1
				speaker_word_count[words[0]] += 1
				if bigram not in vocab:	# calculate |V|
					vocab[bigram] = 1
					vocab_size += 1
				if bigram in speakers[words[0]]:	# if you haven't seen the word before, initialize word count
					speakers[words[0]][bigram] += 1
				else:
					speakers[words[0]][bigram] = 1
				prev = word
			all_words += 1
			speaker_word_count[words[0]] += 1
			if word not in vocab:	# calculate |V|
				vocab[word] = 1
				vocab_size += 1
			if word in speakers[words[0]]:	# if you haven't seen the word before, initialize word count
				speakers[words[0]][word] += 1
			else:
				speakers[words[0]][word] = 1
			just_no = False
			
	###############################
	### Calculate probabilities ###
	###############################
	speaker_probs = {}
	word_probs = {}
	for speaker in speakers:	
		speaker_probs[speaker] = speaker_doc_count[speaker]/all_docs	# calculate p(k)
		word_probs[speaker] = {}
		for word in speakers[speaker]:
			word_probs[speaker][word] = speakers[speaker][word]/speaker_word_count[speaker]	# calculate p(w|k)
	print(speaker_word_count['trump'])
	# 1a
	print('1a.')
	print('total doc count for trump: ', speaker_doc_count['trump'])
	print('total doc count for clinton: ', speaker_doc_count['clinton'])
	print('president word count for trump: ', speakers['trump']['president'])
	print('president word count for clinton: ', speakers['clinton']['president'])
	print('country word count for trump: ', speakers['trump']['country'])
	print('country word count for clinton: ', speakers['clinton']['country'])
	
	# 1b
	print('\n1b.')
	print('probability for trump: ', speaker_probs['trump'])
	print('probability for clinton: ', speaker_probs['clinton'])
	print('president probability for trump: ', word_probs['trump']['president'])
	print('president probability for clinton: ', word_probs['clinton']['president'])
	print('country probability for trump: ', word_probs['trump']['country'])
	print('country probability for clinton: ', word_probs['clinton']['country'])

	# 1c
	devname = 'data/dev'	# read in training data
	# P_k_d = naive_bayes_prob(devname, speakers)
	devlines = open(devname).read().splitlines()
	P_k_d = {}
	for d in devlines: 	# iterate over test documents (d)
		P_k_d[d] = {}
		p_d_k = {}	# init p(d|k) = p(line | speaker)
		p_kd = {}	# init p(k,d)
		for k in speakers:	#iterate over speakers (k)
			words = d.split()		# split by white space into words
			all_ngrams = words[1:]
			p_d_k[k] = 0
			if flag == "suffixes":
				suffix = []
				for word in words[1:]:
					if len(word)>3:
						suffix.append(word[-3:])
					else:
						suffix.append(word)
				all_ngrams.extend(suffix)
			bigrams = []
			if flag == "bigrams":
				if len(words) > 1:	# at least one word in the doc, add start and end tag bigrams
					first = "<s> " + words[1]
					bigrams.append(first)
					end = words[-1] + " </s>"
					bigrams.append(end)
					prev = words[1]
				if len(words) > 2:	# at least 2 words in the doc, iterate over word pairs to make bigrams
					for word in words[2:]:	# iterate over each word in a document
						bigram = prev + " " + word
						bigrams.append(bigram)
						prev = word
				all_ngrams.extend(bigrams)
			for w in all_ngrams:
				delta = 0.1
				if w in speakers[k]:
					p_d_k[k] += np.log((speakers[k][w]+delta))-np.log((speaker_word_count[k]+(vocab_size+1)*delta))	# c(w,k')+delta / sum_w'(c(w',k)) + delta*|v|
				else:
					p_d_k[k] += np.log((delta))-np.log((speaker_word_count[k]+(vocab_size+1)*delta))	# c(w,k')+1 / sum_w'(c(w',k)) + |v|
			p_kd[k] = np.log(speaker_probs[k])+p_d_k[k]	# log(p(k)*p(d|k)
		second = -1*autograd.scipy.misc.logsumexp(list(p_kd.values()))		# second half of equation
		for k in speakers:	#iterate over speakers (k)
			first = p_kd[k]
			P_k_d[d][k] = np.exp(first+second)
	print('\n1c.')
	for k in speakers:
		print(k, ': ', P_k_d[devlines[0]][k])
	#1d
	testname = 'data/test'	# read in training data
	testlines = open(testname).read().splitlines()
	P_k_d = {}
	for d in testlines: 	# iterate over test documents (d)
		P_k_d[d] = {}
		p_d_k = {}	# init p(d|k) = p(line | speaker)
		p_kd = {}	# init p(k,d)
		for k in speakers:	#iterate over speakers (k)
			words = d.split()		# split by white space into words
			p_d_k[k] = 0
			all_ngrams = words[1:]
			if flag == "suffixes":
				suffix = []
				for word in words[1:]:
					if len(word)>3:
						suffix.append(word[-3:])
					else:
						suffix.append(word)
				all_ngrams.extend(suffix)
			bigrams = []
			if flag == "bigrams":
				if len(words) > 1:	# at least one word in the doc, add start and end tag bigrams
					first = "<s> " + words[1]
					bigrams.append(first)
					end = words[-1] + " </s>"
					bigrams.append(end)
					prev = words[1]
				if len(words) > 2:	# at least 2 words in the doc, iterate over word pairs to make bigrams
					for word in words[2:]:	# iterate over each word in a document
						bigram = prev + " " + word
						bigrams.append(bigram)
						prev = word
				all_ngrams.extend(bigrams)
			for w in all_ngrams:
				delta = 0.1
				if w in speakers[k]:
					p_d_k[k] += np.log((speakers[k][w]+delta))-np.log((speaker_word_count[k]+(vocab_size+1)*delta))	# c(w,k')+delta / sum_w'(c(w',k)) + delta*|v|
				else:	# out of vocab
					p_d_k[k] += np.log((delta))-np.log((speaker_word_count[k]+(vocab_size+1)*delta))	# c(w,k')+1 / sum_w'(c(w',k)) + |v|
			p_kd[k] = np.log(speaker_probs[k])+p_d_k[k]	# log(p(k)*p(d|k)
		second = -1*autograd.scipy.misc.logsumexp(list(p_kd.values()))		# second half of equation
		for k in speakers:	#iterate over speakers (k)
			first = p_kd[k]
			P_k_d[d][k] = np.exp(first+second)
	t = 0
	c = 0
	for d in testlines:
		if d.split()[0] == max_prob(P_k_d[d]):
			c+=1
		t+=1
	print('\n1d.')
	print('accuracy: ', c/t)

def max_prob(P_k_d):
	max_speaker = None
	max_prob = 0
	for speaker in P_k_d:
		if P_k_d[speaker] > max_prob:
			max_speaker = speaker
			max_prob = P_k_d[speaker]
	return max_speaker

if __name__ == '__main__':
	main()
