# -*- coding:utf-8 -*-
import nltk
import pickle
import itertools
import sklearn
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
#特征提取方法
#1.将所有词作为特征
def bag_of_words(words):
    return dict([(word,True) for word in words])
#2.把双词搭配（bigrams）作为特征
def bigram(words,score_fn=BigramAssocMeasures.chi_sq,n=1000):
    bigram_finder=BigramCollocationFinder.from_words(words)#把文本变成双词搭配的形式
    bigrams = bigram_finder.nbest(score_fn, n)  # 使用了卡方统计的方法，选择排名前1000的双词
    return bag_of_words(bigrams)
#3.把所有词与双词搭配一起作为特征
def bigram_words(words, score_fn=BigramAssocMeasures.chi_sq, n=1000):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bag_of_words(words + bigrams)  #所有词和（信息量大的）双词搭配一起作为特征

#特征选择方法
'''
1. 计算出整个语料里面每个词的信息量
1.1 计算整个语料里面每个词的信息量
'''
from nltk.probability import FreqDist, ConditionalFreqDist
def create_word_scores():
    posWords = pickle.load(open('weibofile/pos_review.pkl', 'r'))
    negWords = pickle.load(open('weibofile/neg_review1.pkl', 'r'))

    posWords = list(itertools.chain(*posWords))  # 把多维数组解链成一维数组
    negWords = list(itertools.chain(*negWords))

    word_fd = FreqDist()  # 可统计所有词的词频
    cond_word_fd = ConditionalFreqDist()  # 可统计积极文本中的词频和消极文本中的词频
    for word in posWords:
        word_fd[word]+=1
        cond_word_fd['pos'][word]+=1
    for word in negWords:
        word_fd[word] += 1
        cond_word_fd['neg'][word]+=1

    pos_word_count = cond_word_fd['pos'].N()  # 积极词的数量
    neg_word_count = cond_word_fd['neg'].N()  # 消极词的数量
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count),
                                               total_word_count)  # 计算积极词的卡方统计量，这里也可以计算互信息等其它统计量
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count),
                                               total_word_count)  # 同理
        word_scores[word] = pos_score + neg_score  # 一个词的信息量等于积极卡方统计量加上消极卡方统计量

    return word_scores  # 包括了每个词和这个词的信息量
'''
1.2 计算整个语料里面每个词和双词搭配的信息量
'''


def create_word_bigram_scores():
    posdata = pickle.load(open('weibofile/pos_review.pkl', 'r'))
    negdata = pickle.load(open('weibofile/neg_review1.pkl', 'r'))

    posWords = list(itertools.chain(*posdata))
    negWords = list(itertools.chain(*negdata))

    pos_bigram_finder = BigramCollocationFinder.from_words(posWords)
    neg_bigram_finder = BigramCollocationFinder.from_words(negWords)
    posBigrams = pos_bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5000)
    negBigrams = neg_bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5000)

    pos = posWords + posBigrams  # 词和双词搭配
    neg = negWords + negBigrams

    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in pos:
        word_fd[word] += 1
        cond_word_fd['pos'][word]+=1
    for word in neg:
        word_fd[word] += 1
        cond_word_fd['neg'][word]+=1

    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores

'''
2.根据信息量进行倒序排序，选择排名靠前的信息量的词
对find_best_scores赋值
word_scores_1 = create_word_scores()
word_scores_2 = create_word_bigram_scores()
'''
def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number] #把词按信息量倒序排序。number是特征的维度，是可以不断调整直至最优的
    best_words = set([w for w, s in best_vals])
    return best_words

'''
3. 把选出的这些词作为特征（这就是选择了信息量丰富的特征）
'''
def best_word_features(words,best_words):
    return dict([(word, True) for word in words if word in best_words])

# '''
# 整合为使用卡方统计量（Chi-square）来选择信息量丰富的所有词特征的选择方法
# '''
def feature_extraction_method1(words):
    word_scores_1 = create_word_scores()
    best_words=find_best_words(word_scores_1, number=1000)
    return best_word_features(words,best_words)
# '''
# 整合为使用卡方统计量（Chi-square）来选择信息量丰富的所有词+双词特征的选择方法
# '''
def feature_extraction_method2(words):
    word_scores_2 = create_word_bigram_scores()
    best_words=find_best_words(word_scores_2, number=1000)
    return best_word_features(words,best_words)
'''
赋予类标签
pos/neg 格式为：[[word11, word12, ... word1n], [word21, word22, ... , word2n], ... , [wordn1, wordn2, ... , wordnn]]
'''
def pos_features(feature_extraction_method,pos):
    posFeatures = []
    x=1
    for i in pos:
        try:
          posWords = [feature_extraction_method(i),'pos'] #为积极文本赋予"pos"
        except:pass
        posFeatures.append(posWords)
        print x
        x=x+1
    return posFeatures

def neg_features(feature_extraction_method,neg):
    negFeatures = []
    for j in neg:
        try:
           negWords = [feature_extraction_method(j),'neg'] #为消极文本赋予"neg"
        except:pass
        negFeatures.append(negWords)
    return negFeatures