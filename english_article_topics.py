#!/usr/bin/env python
# coding=utf-8
'''这个程序将来自chinadaily的一些英语文章进行话题提取
此外，还有另外一个程序，将对应的中文文章进行话题提取'''

import sys
sys.path.append('../../smt/')
import glob
from preprocessing import fileload,preprocess
import nltk
from gensim import corpora,models
import logging


def corpus_generation():
    '''将原始的爬虫数据转换成篇章语料库
        返回的是一个list数据，list里的每个元素都是
        一个没有标题的已分词的文章内容，未进行深加工
    '''
    
    filelist = glob.glob('../../web_spider/bilingual_article/正常语料/[0-9]*-*')
    english_corpus = []
    english_corpus_article_tokens = []
    english_corpus_tokens = []
    for file in filelist:
        text = fileload(file)
        chinese,english = preprocess(text)
        english_corpus.extend([english[1:]])
    for article in english_corpus:
        for sentence in article:
            english_corpus_article_tokens.extend(nltk.word_tokenize(sentence))
        english_corpus_tokens.append(english_corpus_article_tokens)
        english_corpus_article_tokens = []
    return english_corpus_tokens


def stopwords_load():
    f = open('../stop-words_english_en.txt')
    stopwords = f.read()
    f.close()
    stopwords = set(stopwords.split('\n'))
    return stopwords

def dictionary_generation(corpus,stopwords):
    '''返回字典和再一次处理过的语料（去掉停用词和低频词）, 以矩阵的形式保存
    经第一次执行，发现依然有很多的噪音没有去除，比如数字、符号、和分词产生的噪音
    说明stopwords列表并不全'''
    #去除停用词,这些语句其实可以用函数式编程简便完成
    for article in corpus:
        for num,word in enumerate(article):
            if word in stopwords:
                del article[num]
    
    #去除只出现一次的词
    tokens = []
    for text in corpus:
        tokens.extend(text)
    fdist = nltk.FreqDist(tokens)
    for article in corpus:
        for num,word in enumerate(article):
            if fdist[word] ==1:
                del article[num]
    
    #生成字典和文档矩阵
    dictionary = corpora.Dictionary(corpus)
    filtered_corpus = [dictionary.doc2bow(article) for article in corpus]

    return dictionary,filtered_corpus


def topic_extraction(corpus,dict):
    '''话题提取'''
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lsi = models.LsiModel(corpus_tfidf,id2word=dict,num_topics=30)
    corpus_lsi = lsi[corpus_tfidf]
    lsi.print_topics(10)
    for doc in corpus_lsi[0:10]:
        print(doc)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    dict,corpus = dictionary_generation(corpus_generation(),stopwords_load())
    
    dict.save('english_corpus_dictionary.dict')
    corpora.MmCorpus.serialize('english_corpus_matrix.mm',corpus)
    topic_extraction(corpus,dict)
    print('done')

