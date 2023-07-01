# Imports
import nltk
import spacy
import string, random, re, os #sklearn# contractions, matplotlib,
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

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

import random
import functools
import spacy

# Logistic Regression Packages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score

# Neural Network Packages
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Random Forest Packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

NER = spacy.load("en_core_web_sm")

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# 1 if Comp Authoritarian
# 0 if Democ
Regime = pd.DataFrame({'name': ['Australia_Constitution.txt',
					'China_Constitution.txt',
					'Egypt_Constitution.txt',
					'France_Constitution.txt',
					'Germany_Constitution.txt',
					'Japan_Constitution.txt',
					'Russia_Constitution.txt',
					'Turkey_Constitution.txt',
					'US_Constitution.txt',
					'Venezuela_Constitution.txt',
					'Belarus_Constitution.txt',
					'Cambodia_Constitution.txt',
					'Canada_Constitution.txt',
					'Chile_Constitution.txt',
					'Iran_Constitution.txt',
					'Kazakhstan_Constitution.txt',
					'Nicaragua_Constitution.txt',
					'Portugal_Constitution.txt',
					'South_Korea_Constitution.txt',
					'Sweden_Constitution.txt'],
                    'regime': [0, 1, 1, 0, 0, 0, 1, 1, 0,
			       			1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0]})


def main():
	print('NLP corpus reader')
	# Call functions to run examples

	corpus_reader = corpus_reader_plain_example()
	# corpus_reader = corpus_reader_categorized_example_manual()
	print()
	new_counts = corpus_counts(corpus_reader)
	new_content = nlp_pipe(corpus_reader)
	Main_Data = pd.merge(new_counts, new_content, on='name')
	Main_Data = pd.merge(Main_Data, Regime, on='name')
	print(Main_Data)
	print()
	Log_Reg(Main_Data)
	print()
	NN_Classifier(Main_Data)
	print()
	RF_Classifier(Main_Data)




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
					'Venezuela_Constitution.txt',
					'Belarus_Constitution.txt',
					'Cambodia_Constitution.txt',
					'Canada_Constitution.txt',
					'Chile_Constitution.txt',
					'Iran_Constitution.txt',
					'Kazakhstan_Constitution.txt',
					'Nicaragua_Constitution.txt',
					'Portugal_Constitution.txt',
					'South_Korea_Constitution.txt',
					'Sweden_Constitution.txt'])
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
		# Standardizing them based on the number of characters
		num_words = num_words / num_chars
		num_sents = num_sents / num_chars
		num_vocab = num_vocab / num_chars
		num_chars = num_chars / 100000
		# frequency_distribution = FreqDist(corpus_reader.words(fileids))
		file_counts = np.array([fileids, num_chars, num_words, num_sents, num_vocab])
		doc_counts = np.append(doc_counts, [file_counts], axis = 0)
		Counts_variables = pd.DataFrame(doc_counts[1:], columns=doc_counts[0])
	print(Counts_variables)
	return Counts_variables


def nlp_pipe(corpus_reader):

	print('\n\n')
	stop_words = stopwords.words('english')
	stop_words = [
		'|', '&', '!', '@', '#', '$', '%', '*', '(', ')', '-', '_', "'", ";", ":",
		"'", ".",",","1","2","3","4","5","6","7","8","9","0"
	] + stop_words
	porter_stemmer = PorterStemmer()
	word_net_lemmatizer = WordNetLemmatizer()
	
	NER_vars = np.array([['name', 'norp_count', 'org_count', 'person_count', 'gpe_count']])

	# Construct the NLP pipeline
	for fileids in corpus_reader.fileids():
		content = [
			w.lower() for w in corpus_reader.words(fileids)
			if w.lower() not in stop_words
		]
		#content = [contractions.fix(word) for word in content]  # expand contractions
		content = [porter_stemmer.stem(word) for word in content]
		content = [word_net_lemmatizer.lemmatize(word) for word in content]

		# print(content)
		NER_scores = Get_NER(content)
		NER_scores.insert(0, fileids)
		NER_vars = np.append(NER_vars, [NER_scores], axis = 0)
		# print(NER_vars)

	NER_variables = pd.DataFrame(NER_vars[1:], columns=NER_vars[0])
	print(NER_variables)
	return NER_variables

# Feature Extraction
# Bigram, Trigram, Quadgram, NER group,

# NER
# Get_NER(content):
def Get_NER(content):
	norp_count = 0
	org_count = 0
	person_count = 0
	gpe_count = 0
	event_count = 0
	for word in content:
		doc = NER(word)
		for ent in doc.ents:
				label = ent.label_
				if label == "NORP":
					norp_count += 1
				elif label == "ORG":
					org_count += 1
				elif label == "PERSON":
					person_count += 1
				elif label == "GPE":
					gpe_count += 1

	NER_Total = norp_count + org_count + person_count + gpe_count + event_count
	norp_count_stand = norp_count / NER_Total 
	norp_count_stand = round(norp_count_stand, 3)
	org_count_stand = org_count / NER_Total 
	org_count_stand = round(org_count_stand, 3)
	person_count_stand = person_count / NER_Total 
	person_count_stand = round(person_count_stand, 3)
	gpe_count_stand = gpe_count / NER_Total 
	gpe_count_stand = round(gpe_count_stand, 3)

	NER_Scores = [norp_count_stand, org_count_stand, person_count_stand, gpe_count_stand]
	return NER_Scores
	
# The NN pipeline


# Logistic Regression
def Log_Reg(Main_Data):
	# ID DV and IV Vars
	X = Main_Data[['num_chars', 'num_words', 'num_sents', 'num_vocab', 'norp_count', 'org_count', 'person_count', 'gpe_count']]  # Input features
	y = Main_Data['regime']
	# Create Test, Train Split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	# Define Logreg
	logreg = LogisticRegression()
	# Fit the model
	logreg.fit(X_train, y_train)
	# Predict Y with fitted model
	y_pred = logreg.predict(X_test)
	# Print classification table of prediction vs test outcome
	print(classification_report(y_test, y_pred))

	accuracy = accuracy_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred)
	precision = precision_score(y_test, y_pred)
	print("Logistic Regression:")
	print("Accuracy:", accuracy)
	print("Recall:", recall)
	print("Precision:", precision)


# Neural Network Classifier
def NN_Classifier(Main_Data):
	# ID DV and IV Vars
	X = Main_Data[['num_chars', 'num_words', 'num_sents', 'num_vocab', 'norp_count', 'org_count', 'person_count', 'gpe_count']]  # Input features
	y = Main_Data['regime']

	# Split the data into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Create and train a multi-layer perceptron (MLP) classifier
	clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42)
	clf.fit(X_train, y_train)

	# Predict the test set labels
	y_pred = clf.predict(X_test)

	# Create a confusion matrix
	cm = confusion_matrix(y_test, y_pred)

	# Create a DataFrame to display the confusion matrix
	cm_df = pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_)

	# Plot the confusion matrix
	plt.figure(figsize=(8, 6))
	sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='d')
	plt.title('Confusion Matrix')
	plt.xlabel('Predicted Label')
	plt.ylabel('True Label')
	plt.show()

	accuracy = accuracy_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred)
	precision = precision_score(y_test, y_pred)
	print("Neural Network:")
	print("Accuracy:", accuracy)
	print("Recall:", recall)
	print("Precision:", precision)



# Random Forest
def RF_Classifier(Main_Data):
	# ID DV and IV Vars
	X = Main_Data[['num_chars', 'num_words', 'num_sents', 'num_vocab', 'norp_count', 'org_count', 'person_count', 'gpe_count']]  # Input features
	y = Main_Data['regime']

	# Split the data into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Create a Random Forest classifier
	rtf = RandomForestClassifier(n_estimators=100, random_state=42)

	# Train the classifier
	rtf.fit(X_train, y_train)

	# Predict the test set labels
	y_pred = rtf.predict(X_test)

	# Get the confusion matrix
	rf_cm = confusion_matrix(y_test, y_pred)

	# Create a DataFrame to display the confusion matrix
	rf_cm_df = pd.DataFrame(rf_cm, index=rtf.classes_, columns=rtf.classes_)

	# Plot the confusion matrix
	plt.figure(figsize=(8, 6))
	sns.heatmap(rf_cm_df, annot=True, cmap='Blues', fmt='d')
	plt.title('Confusion Matrix')
	plt.xlabel('Predicted Label')
	plt.ylabel('True Label')
	plt.show()

	accuracy = accuracy_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred)
	precision = precision_score(y_test, y_pred)
	print("Random Forest:")
	print("Accuracy:", accuracy)
	print("Recall:", recall)
	print("Precision:", precision)

	# Get feature importance
	importance = rtf.feature_importances_

	# Sort feature importance in descending order
	sorted_indices = importance.argsort()[::-1]
	sorted_importance = importance[sorted_indices]

	# Create a list of feature names
	feature_names = ['num_chars', 'num_words', 'num_sents', 'num_vocab', 'norp_count', 'org_count', 'person_count', 'gpe_count']

	# Plot feature importance
	plt.figure(figsize=(10, 6))
	plt.bar(range(len(sorted_importance)), sorted_importance)
	plt.xticks(range(len(sorted_importance)), [feature_names[i] for i in sorted_indices], rotation='vertical')
	plt.xlabel('Feature')
	plt.ylabel('Importance')
	plt.title('Random Forest Feature Importance')
	plt.tight_layout()
	plt.show()


if __name__ == '__main__':
	main()
