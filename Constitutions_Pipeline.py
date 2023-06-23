# Imports
import nltk
import spacy
import string, random, re, os #sklearn# contractions, matplotlib,
import numpy as np
import pandas as pd

from nltk import word_tokenize
from nltk import bigrams, trigrams, ngrams

from nltk.corpus.reader import PlaintextCorpusReader
from nltk.corpus.reader import CategorizedPlaintextCorpusReader
from nltk.corpus import stopwords

from nltk.probability import FreqDist

from collections import Counter
from collections import defaultdict

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# from sklearn import CountVectorizer

import random
import functools

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


def main():
	print('NLP corpus reader')
	# Call functions to run examples

	corpus_reader = corpus_reader_plain_example()
	# corpus_reader = corpus_reader_categorized_example_manual()
	print()
	corpus_counts(corpus_reader)
	print()
	new_content = nlp_pipe(corpus_reader)
	# print(new_content)

def corpus_reader_plain_example():
	corpus_reader = PlaintextCorpusReader('./Corpus',
					['Australia_Constitution.txt',
					'China_Constitution.txt',
					'Egypt_Constitution.txt',
					'France_Constitution.txt',
					'Germany_Constitution.txt',
					'Japan_Constitution.txt',
					'Russia_Constitution.txt',
					'Turkey_Constitution.txt',
					'US_Constitution.txt',
					'Venezuela_Constitution.txt'])
	# for document in corpus_reader.fileids():
	#	print(document)
	return corpus_reader


'''
def corpus_reader_categorized_example_manual():
	corpus_reader = CategorizedPlaintextCorpusReader('./Corpus', r'.*\_.*\.txt',
										cat_map={'./Corpus/Australia_Constitution.txt': ['AU_TXT'],
										'./Corpus/China_Constitution.txt': ['CN_TXT'],
										'./Corpus/Egypt_Constitution.txt': ['EG_TXT'],
										'./Corpus/France_Constitution.txt': ['FR_TXT'],
										'./Corpus/Germany_Constitution.txt': ['GM_TXT'],
										'./Corpus/Japan_Constitution.txt': ['JP_TXT'],
										'./Corpus/Russia_Constitution.txt': ['RU_TXT'],
										'./Corpus/Turkey_Constitution.txt': ['TR_TXT'],
										'./Corpus/US_Constitution.txt': ['US_TXT'],
										'./Corpus/Venezuela_Constitution.txt': ['VE_TXT'],
										})
	for category in corpus_reader.categories():
			print(category)
	print(corpus_reader.fileids('AU_TXT'))
	print(corpus_reader.fileids('CN_TXT'))
	print(corpus_reader.fileids('EG_TXT'))
	print(corpus_reader.fileids('FR_TXT'))
	print(corpus_reader.fileids('GM_TXT'))
	print(corpus_reader.fileids('JP_TXT'))
	print(corpus_reader.fileids('RU_TXT'))
	print(corpus_reader.fileids('TR_TXT'))
	print(corpus_reader.fileids('US_TXT'))
	print(corpus_reader.fileids('VE_TXT'))
	return corpus_reader
'''


def corpus_counts(corpus_reader):
	doc_counts = np.array([['name','num_chars', 'num_words', 'num_sents', 'num_vocab']])
	for fileids in corpus_reader.fileids():
		num_chars = len(corpus_reader.raw(fileids))
		num_words = len(corpus_reader.words(fileids))
		num_sents = len(corpus_reader.sents(fileids))
		num_vocab = len(set([w.lower() for w in corpus_reader.words(fileids)]))
		# frequency_distribution = FreqDist(corpus_reader.words(fileids))
		file_counts = np.array([fileids, num_chars, num_words, num_sents, num_vocab])
		doc_counts = np.append(doc_counts, [file_counts], axis = 0)

		print(doc_counts)
		return doc_counts


def nlp_pipe(corpus_reader):
	print('\n\n')
	stop_words = stopwords.words('english')
	stop_words = [
		'|', '&', '!', '@', '#', '$', '%', '*', '(', ')', '-', '_', "'", ";", ":",
		"'", ".",",","1","2","3","4","5","6","7","8","9","0"
	] + stop_words
	porter_stemmer = PorterStemmer()
	word_net_lemmatizer = WordNetLemmatizer()

	BOW_Matrix = [['test','australia',1]]
	BOW_Matrix = pd.DataFrame(BOW_Matrix)
	BOW_Matrix.columns = ['name', 'word','value']


	# Construct the NLP pipeline
	for fileids in corpus_reader.fileids():
		content = [
			w.lower() for w in corpus_reader.words(fileids)
			if w.lower() not in stop_words
		]
		#content = [contractions.fix(word) for word in content]  # expand contractions
		content = [porter_stemmer.stem(word) for word in content]
		content = [word_net_lemmatizer.lemmatize(word) for word in content]
				
		Top_BOW = Bag_of_Words(content)
		Top_BOW = np.array(Top_BOW)
		# Top_BOW = list(map(lambda i: (fileids, i), Top_BOW))
		Top_BOW = pd.DataFrame(Top_BOW)
		Top_BOW.columns = ['word', 'freq']
		Top_BOW = Top_BOW.drop('freq', axis = 1)	
		file_name = [fileids] * 20
		value = [1] * 20
		Top_BOW.insert(0, "name", file_name)	
		Top_BOW.insert(1, "value", value)	
		# Top_BOW = Top_BOW.pivot(index = 'name', columns='word', values = 'value')
		BOW_Matrix = pd.merge(BOW_Matrix, Top_BOW, on = ['name','word','value'], how ='outer')
		BOW_Matrix = BOW_Matrix[BOW_Matrix.name != 'test']

		# BOW_Matrix = BOW_Matrix.pivot(index ='name', columns='word', values = 'value')

		
		print(BOW_Matrix)
		# print(Top_BOW)
		print()
		Top_Bigrams = Get_Bigrams(content)
		Top_Bigrams = list(map(lambda i: (fileids, i), Top_Bigrams))
		# print(Top_Bigrams)
		print()
		Top_Trigrams = Get_Trigrams(content)
		Top_Trigrams = list(map(lambda i: (fileids, i), Top_Trigrams))
		# print(Top_Trigrams)
		print()
		Top_Quadgrams = Get_Quadgrams(content)
		Top_Quadgrams = list(map(lambda i: (fileids, i), Top_Quadgrams))
		# print(Top_Quadgrams)
		print()

		# print(content)
		#  Get_NER(content)

# Feature Extraction
# Bigram, Trigram, Quadgram, NER group,


# Bag of Words
def Bag_of_Words(content):
		counts = Counter(content)
		# IDs the most common 20 words for each fileid
		Top_words = counts.most_common(n=20)
		# Prints the most common 20 words for each fileid
		# print(Top_words)
		return Top_words


# Bigrams
def Get_Bigrams(content):
	bigrams_list = list(bigrams(content, pad_left=True, pad_right=True))
	bigrams_count = Counter(bigrams_list)
	# IDs the most common 20 words for each fileid
	top_bigrams = bigrams_count.most_common(n=20)
	# Prints the most common 20 words for each fileid
	# print(top_bigrams)
	return top_bigrams

# Trigrams
def Get_Trigrams(content):
	trigrams_list = list(trigrams(content, pad_left=True, pad_right=True))
	trigrams_count = Counter(trigrams_list)
	# IDs the most common 20 words for each fileid
	top_trigrams = trigrams_count.most_common(n=20)
	# Prints the most common 20 words for each fileid
	# print(top_trigrams)
	return top_trigrams


# Quadgrams
# Get_QuadGrams(content):
def Get_Quadgrams(content):
	quadgrams_list = (list(ngrams(content, 4, pad_left=True, pad_right=True)))
	quadgrams_count = Counter(quadgrams_list)
	# IDs the most common 20 words for each fileid
	top_quadgrams = quadgrams_count.most_common(n=20)
	# Prints the most common 20 words for each fileid
	# print(top_quadgrams)
	return top_quadgrams


# NER
# Get_NER(content):
def Get_NER(content):
	for ent in content.ents:
		NER_labels = (ent.label_)
		if NER_labels == "NORP":
			norp_count = +1
		if NER_labels == "ORG":
			org_count = +1
		if NER_labels == "PERSON":
			person_count = +1
		return norp_count, org_count, person_count,


if __name__ == '__main__':
	main()
