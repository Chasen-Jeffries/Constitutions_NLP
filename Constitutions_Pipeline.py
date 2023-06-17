# Imports
import nltk
import spacy
import string, random, re, contractions, matplotlib, os, sklearn 

from nltk import word_tokenize
from nltk import bigrams, trigrams, ngrams


from nltk.corpus.reader import PlaintextCorpusReader
from nltk.corpus.reader import CategorizedPlaintextCorpusReader
from nltk.corpus import stopwords
from nltk.corpus import reuters
from nltk.probability import FreqDist

from collections import Counter

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# from sklearn import CountVectorizer

import random
import functools

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('reuters')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


def main():
  print('NLP corpus reader')
  # Call functions to run examples

  # corpus_reader = corpus_reader_simple_example()
  # corpus_counts(corpus_reader)
  # nlp_pipe(corpus_reader)

  corpus_reader = corpus_reader_categorized_example_manual()
  corpus_counts(corpus_reader)
  nlp_pipe(corpus_reader)

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
  for fileid in corpus_reader:
    num_chars = len(corpus_reader.raw(fileid))
    num_words = len(corpus_reader.words(fileid))
    num_sents = len(corpus_reader.sents(fileid))
    num_vocab = len(set([w.lower() for w in corpus_reader.words(fileid)]))
    frequency_distribution = FreqDist(corpus_reader.words(fileid))

    # how do I make the output a matrix of values rather than this printout?
    return num_chars, num_words, num_sents, num_vocab, frequency_distribution


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
      content = [word_tokenize(word).tag_ for word in content] # Will this work with Spacy POS tag
      content = [content.str.replace('[^\w\s]','')]     # Remove punctuation, or do I not want this yet?      

      Bag_of_Words(content)
      Get_Bigrams(content)
      Get_Trigrams(content)
      Get_Quadgrams(content)
      Get_POSCounts(content)    
      Get_NER(content)

# Feature Extraction
# Bigram, Trigram, Quadgram, NER group, 

# Bag of Words
def Bag_of_Words(content):
  counts = Counter(content)
    # total_count = len(reuters.words())
      # IDs the most common 20 words for each fileid
  Top_words = counts.most_common(n=20)
      # Prints the most common 20 words for each fileid
    # print('\n\n')
    # print(Top_words) 
  return Top_words
    # Compute the Relative Frequencies
    # for word in counts:
    #  rel_freq = counts[word] / float(total_count)  # Why does it not work when I try "T = counts[word] /= float(total_count)"
    #  print(rel_freq)

# Bigrams
def Get_Bigrams(content):
  bigrams_list = list(bigrams(content, pad_left=True, pad_right=True))
    # IDs the most common 20 words for each fileid
  top_bigrams = bigrams_list.most_common(n=20) 
    # Prints the most common 20 words for each fileid
    # print(top_bigrams) 
  return top_bigrams 

# Trigrams
def Get_Trigrams(content):
  trigrams_list = list(trigrams(content, pad_left=True, pad_right=True))
    # IDs the most common 20 words for each fileid
  top_trigrams = trigrams_list.most_common(n=20) 
    # Prints the most common 20 words for each fileid
    # print(top_trigrams) 
  return top_trigrams 

# Quadgrams
# Get_QuadGrams(content):
def Get_Quadgrams(content):
  quadgrams_list = (list(ngrams(content, 4, pad_left=True, pad_right=True)))
    # IDs the most common 20 words for each fileid
  top_quadgrams = quadgrams_list.most_common(n=20) 
    # Prints the most common 20 words for each fileid
    # print(top_quadgrams) 
  return top_quadgrams 

# POS Counts
# Get_POSCounts(content)
def Get_POSCounts(content):
  num_pos = content.count_by(spacy.attrs.POS)
  for k,v in sorted(num_pos.items()):
    pos_counts = (f'{k}. {content.vocab[k].text:{8}}: {v}')
    return pos_counts


# NER 
# Get_NER(content):
def Get_NER(content):
    for ent in content.ents:
      NER_labels = (ent.label_)    
      if NER_labels == "NORP":
         norp_count =+ 1
      if NER_labels == "ORG":
         org_count =+ 1
      if NER_labels == "PERSON":
         person_count =+ 1
      return norp_count, org_count, person_count, 
    
      


if __name__ == '__main__':
  main()