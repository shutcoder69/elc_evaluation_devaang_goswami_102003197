from nltk.stem import PorterStemmer
import math
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

ps=PorterStemmer()
stopwords=list(set(stopwords.words('english')))

def create_frequency_matrix(sentences):
    frequency_matrix = {}

    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
        
            if word in freq_table and word not in stopwords:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent] = freq_table

    return frequency_matrix


def create_tf_matrix(freq_matrix):
    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix


def create_documents_per_words(freq_matrix):
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
          # occureance of word in sentences
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table


def create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
  # total_documents=total_sentences
  # inverse document frequency
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix


def create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix


def score_sent(tf_idf_matrix):

    sentenceValue = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += 0.5*score

        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

    return sentenceValue


def tfidf(sent_tokens):
    sentences = sent_tokens
    total_documents = len(sentences)

    freq_matrix = create_frequency_matrix(sentences)
    #print(freq_matrix)

    tf_matrix = create_tf_matrix(freq_matrix)
    #print(tf_matrix)

    count_doc_per_words = create_documents_per_words(freq_matrix)
    #print(count_doc_per_words)

    idf_matrix = create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
    
    tf_idf_matrix = create_tf_idf_matrix(tf_matrix, idf_matrix)

    tfidf_score = score_sent(tf_idf_matrix)

    # print(tfidf_score.values())
    return tfidf_score