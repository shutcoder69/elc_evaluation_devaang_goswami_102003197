import streamlit as st
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import nltk 
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
import regex as re
import math
from features import sentence_score,cue_phrase_cal,upper_cal,digit_cal,sentence_pos,sentence_len_cal,pnoun_cal,heading_cal,text_cleaner,steming
from tfidf import tfidf
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import *

st.title("☂€✠☂")
st.title("Text Summarization App ✍")
st.markdown("--- made by Devaang Goswami")


text = st.text_area('Write your paragraph here:') 

if st.button('SUMMARIZE'):
    # preprocessing 
    text=text_cleaner(text)
    # print(text)
    sent_tokens=sent_tokenize(text)
    word_tokens=word_tokenize(text)
    word_tokens=[word.lower() for word in word_tokens]

    # removing stopwords
    stopwords=list(set(stopwords.words('english')))
    word_token_ref=[word for word in word_tokens if word not in stopwords]

    word_final=steming(word_token_ref)
    freq={}
    for word in word_final:
      if word in freq:
        freq[word]+=1
      else:
        freq[word]=1

    scores_fun=[sentence_score(sent_tokens,freq),cue_phrase_cal(sent_tokens,word_tokens),upper_cal(sent_tokens,word_tokens),digit_cal(sent_tokens,word_tokens),sentence_pos(sent_tokens),sentence_len_cal(sent_tokens),pnoun_cal(sent_tokens),tfidf(sent_tokens),heading_cal(sent_tokens)]

    total_score={}

    for fun in scores_fun:
      arr=fun
      for sent in sent_tokens:
        total_score[sent]=0
        if sent in arr:
          total_score[sent]+=arr[sent]

    avg=np.mean(list(total_score.values()))

    summary=""
    for sentence in sent_tokens:
      if total_score[sentence]>1.2*avg:
        summary+=sentence+" "
    # print(summary)
    st.success(summary)

    cloud=WordCloud(stopwords=STOPWORDS,background_color="white",height=650,width=800).generate(summary)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # generate to make word cloud of words that are common in particular sentiments
    plt.imshow(cloud)
    plt.axis("off")
    st.pyplot()
    # to hide warning
    
