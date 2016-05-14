#!/usr/bin/env python
# coding=utf-8
'''此程序将chinadaily中的一些中文文章进行话题抽取'''

import sys 
sys.path.append('../../smt/')
import glob
from preprocessing import fileload,preprocess
import jieba
from gensim import corpora,models
import logging

def corpus_generation():

    filelist = glob.glob('../../web_spider/bilingual_article/正常语料/[0-9]*-*')
    chinese_corpus = []
    chinese_corpus_article_tokens = []
    chinese_corpus_tokens = []
    for file in filelist:
        text = fileload(file)
        chinese,english = preprocess(text)
        chinese_corpus.extend([chinese[1:]])
    for article in chinese_corpus:
        for sentence in article:
            chinese_corpus_article_tokens.extend(jieba.lcut(sentence,cut_all=False))
        chinese_corpus_tokens.append(chinese_corpus_article_tokens)
        chinese_corpus_article_tokens=[]
    return chinese_corpus_tokens

def stopwords_load():
    f = open('../stop_words_chinese_cn.txt')
    stop_words = f.read()
    stop_words = set(stop_words.split('\n'))
    return stop_words

def dictionary_generation(corpus,stopwords):
    for article in corpus:
        for num,word in enumerate(article):
            if word in stopwords:
                del article[num]

    dictionary = corpora.Dictionary(corpus)
    filtered_corpus = [dictionary.doc2bow(article) for article in corpus]

    return dictionary,filtered_corpus

def topic_extraction(corpus,dict):
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lsi = models.LsiModel(corpus_tfidf,id2word=dict,num_topics=10)
    corpus_lsi = lsi[corpus_tfidf]
    lsi.print_topics(10)
    for doc in corpus_lsi[0:10]:
        print(doc)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level = logging.INFO)
    dict,corpus = dictionary_generation(corpus_generation(),stopwords_load())
    dict.save('chinese_corpus_dictionary.dict')
    corpora.MmCorpus.serialize('chinese_corpus_matrix.mm',corpus)
    topic_extraction(corpus,dict)
    print('done')
