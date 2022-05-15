import regex as re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.tag import pos_tag

import pandas as pd
import numpy as np

ps=PorterStemmer()
stopwords=list(set(stopwords.words('english')))

def text_cleaner(text):
    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]
    for rule in rules:
      for (k, v) in rule.items():
          rx = re.compile(k)
          text = rx.sub(v, text)
      text = text.rstrip()
    return text


def steming(words):
  stem=[]
  # play,played..
  for word in words:
    stem.append(ps.stem(word))
  # print(stem)
  return stem

def stemSentence(sentence):
    sentence=sentence.lower()
    token_words=word_tokenize(sentence)
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(ps.stem(word))
        stem_sentence.append(" ")
    return stem_sentence

# computing the score of each sentence

def sentence_score(sent_tokens,freq):
      sentval={}
      sumval=0
      for sentence in sent_tokens:
        sentval[sentence]=0
        for word, count in freq.items():
          sent=stemSentence(sentence)
          if word in sent:
            sentval[sentence]+=count
      maxi=max(sentval.values())
      for key,val in sentval.items():
        sentval[key]=val/maxi
      # print(sentval.values())
      return sentval

# calculating cue -phrase

def cue_phrase_cal(sent_tokens,word_tokens):
    cue_phrase=['firstly','secondly','thirdly','assumption','anyway','conclusion','nutshell','moreover']
    qphrase={}
    for sent in sent_tokens:
      qphrase[sent]=0
      tokens_word=word_tokenize(sent)
      for word in word_tokens:
        if word.lower() in cue_phrase:
          qphrase[sent]+=1

    maxi=max(qphrase.values())

    for key,val in qphrase.items():
      try:
        qphrase[key]=val/maxi
      except: 
        pass
    # print(qphrase.values())
    return qphrase

# calculating upper case 

def upper_cal(sent_tokens,word_tokens):
    upper={}
    for sent in sent_tokens:
      upper[sent]=0
      tokens_word=word_tokenize(sent)
      for word in word_tokens:
        if word.isupper():
          upper[sent]+=1

    maxi=max(upper.values())

    for key,val in upper.items():
      try:
        upper[key]=val/maxi
      except: 
        pass
    # print(upper.values())
    return upper

# calculating upper case 

def digit_cal(sent_tokens,word_tokens):
    digit={}
    for sent in sent_tokens:
      digit[sent]=0
      tokens_word=word_tokenize(sent)
      for word in word_tokens:
        if word.isdigit():
          digit[sent]+=1

    maxi=max(digit.values())

    for key,val in digit.items():
      try:
        digit[key]=val/maxi
      except: 
        pass
    # print(digit.values())
    return digit

# calculating proper noun 


def pnoun_cal(sent_tokens):
    pnoun={}
    for sent in sent_tokens:
      pnoun[sent]=0
      tagged_sent=pos_tag(sent.split())
      proper=[word for word,tag in tagged_sent if tag=='NNP']
      pnoun[sent]=len(proper)

    maxi=max(pnoun.values())

    for key,val in pnoun.items():
      try:
        pnoun[key]=val/maxi
      except: 
        pass
    # print(pnoun.values())
    return pnoun

def sentence_len_cal(sent_tokens):
    sent_len={}
    for sent in sent_tokens:
      sent_len[sent]=0
      word_tokens=word_tokenize(sent)
      length=len(word_tokens)
      if length in range(0,10):
        sent_len[sent]=1-0.05*(10-length)
      elif length in range(10,20):
        sent_len[sent]=1
      else:
        sent_len[sent]=1-0.05*(length-20)
    # print(sent_len.values())
    return sent_len

def sentence_pos(sent_tokens):
  sentence_position={}
  n=1
  N=len(sent_tokens)
  for sent in sent_tokens:
    a=1/n
    b=1/(N+1-n)
    sentence_position[sent]=max(a,b)
    n=n+1
  # print(sentence_position.values())
  return sentence_position

def heading_cal(sent_tokens):
    head={}
    head_tokens=stemSentence(sent_tokens[0])

    for sent in sent_tokens:
      head[sent]=0
      word_tokens=stemSentence(sent)

      for word in word_tokens:
        if word not in stopwords and word in head_tokens:
          head[sent]+=1

    maxi=max(head.values())

    for key,val in head.items():
      try:
        head[key]=val/maxi
      except: 
        pass
    # print(head.values())
    return head

