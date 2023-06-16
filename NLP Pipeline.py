# Imports
import nltk
import os 
# Importing matplotlib to help with plotting the data
import matplotlib
import contractions
import re

from nltk import word_tokenize
from nltk import bigrams, trigrams

from nltk.corpus.reader import PlaintextCorpusReader
from nltk.corpus.reader import CategorizedPlaintextCorpusReader
from nltk.corpus import stopwords
from nltk.corpus import reuters

from collections import Counter

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from nltk.probability import FreqDist

import random
import functools

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('reuters')
nltk.download('punkt')


def main():
  print('NLP corpus reader')
  # Call functions to run examples

  # corpus_reader = corpus_reader_simple_example()
  # corpus_counts(corpus_reader)
  # nlp_pipe(corpus_reader)

  corpus_reader = corpus_reader_categorized_example_manual()
  corpus_counts(corpus_reader)
  content = nlp_pipe(corpus_reader)
  Bag_of_Words(content)
  Get_Bigrams(content)
  Get_Trigrams(content)

def corpus_reader_categorized_example_manual():
  corpus_reader_cat = CategorizedPlaintextCorpusReader('./Corpus', r'.*\_.*\.txt',
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
  print('****Categories List Manual****')
  print(corpus_reader_cat.categories())
  for category in corpus_reader_cat.categories():
      print(category)
  print()
  print('****Documents by Categories Category****')
  print(corpus_reader_cat.fileids('AU_TXT'))
  print(corpus_reader_cat.fileids('CN_TXT'))
  print(corpus_reader_cat.fileids('EG_TXT'))
  print(corpus_reader_cat.fileids('FR_TXT'))
  print(corpus_reader_cat.fileids('GM_TXT'))
  print(corpus_reader_cat.fileids('JP_TXT'))
  print(corpus_reader_cat.fileids('RU_TXT'))
  print(corpus_reader_cat.fileids('TR_TXT'))
  print(corpus_reader_cat.fileids('US_TXT'))
  print(corpus_reader_cat.fileids('VE_TXT'))
  print('*********************************************')
  return corpus_reader_cat


def corpus_counts(corpus_reader):
  print('chars/word\t', 'words/sent\t', 'words/vocab\t', 'fileids')
  for fileid in corpus_reader.fileids():
    num_chars = len(corpus_reader.raw(fileid))
    num_words = len(corpus_reader.words(fileid))
    num_sents = len(corpus_reader.sents(fileid))
    num_vocab = len(set([w.lower() for w in corpus_reader.words(fileid)]))
    frequency_distribution = FreqDist(corpus_reader.words(fileid))

    # how do I make the output a matrix of values rather than this printout?
    print(num_chars, num_words, format(float(num_chars / num_words),'.3f'), '\t\t', 
          num_words, num_sents, format(float(num_words / num_sents), '.3f'), '\t\t', 
          num_vocab, format(float(num_words / num_vocab), '.3f'), '\t', 
          fileid, '\n',
          frequency_distribution)
    

def nlp_pipe(corpus_reader):
  print('\n\n')
  stop_words = stopwords.words('english')
  stop_words = ['|', '&','!','@','#','$','%','*','(',')','-','_',"'",] + stop_words
  porter_stemmer = PorterStemmer()
  word_net_lemmatizer = WordNetLemmatizer()


# Construct the NLP pipeline
  for fileids in corpus_reader.fileids():
      content = [w.lower() for w in corpus_reader.words(fileids) if w.lower() not in stop_words]
      content = [contractions.fix(word) for word in content]  # expand contractions
      content = [porter_stemmer.stem(word) for word in content]
      content = [word_net_lemmatizer.lemmatize(word) for word in content]
      # content = [content.str.replace('[^\w\s]','')]     # Remove punctuation, or do I not want this yet?      
      content = [nltk.pos_tag(word_tokenize(word)) for word in content]
      print('Final content\n', content)
      print('\n\n\n')
      

    # View corpus as sentences
    #  for sent in corpus_reader.sents():
    #   print(sent)
    #  print('\n\n\n')

# Feature Extraction
# Bigram, Trigram, Quadgram, NER group freq
# TF-IDF at regime type level 

# Bag of Words
def Bag_of_Words(content):
  print('\n\n')
  print('***********Bag of Words Language Model***********')
  counts = Counter(content)
    # total_count = len(reuters.words())
    # IDs the most common 20 words for each fileid
  Top_words = counts.most_common(n=20)
    # Prints the most common 20 words for each fileid
  print('\n\n')
  print(Top_words) 
    
    # Compute the Relative Frequencies
    # for word in counts:
    #  rel_freq = counts[word] / float(total_count)  # Why does it not work when I try "T = counts[word] /= float(total_count)"
    #  print(rel_freq)

# Bigrams
def Get_Bigrams(content):
    bigrams_list = list(bigrams(content))
    # IDs the most common 20 words for each fileid
    top_bigrams = bigrams_list.most_common(n=20) 
    # Prints the most common 20 words for each fileid
    print(top_bigrams, fileid) 


def Get_Trigrams(content):
   




if __name__ == '__main__':
  main()
