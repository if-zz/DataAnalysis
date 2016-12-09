# -*- coding:utf-8 -*-
import sklearn
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import feature
from feature import *
import pickle
from random import shuffle

def score(classifier):
    classifier = SklearnClassifier(classifier) #在nltk 中使用scikit-learn 的接口
    classifier.train(train)  #训练分类器,train是训练集合
    # pickle.dump(classifier,open("./weibofile"+'/'+'classifier'+'.pkl','wb')) #为了方便以后使用，可以将该分类器存储下来
    devSet, tag_dev = zip(*devtest)
    pred = classifier.classify_many(devSet) #对开发测试集的数据进行分类，给出预测的标签
    return accuracy_score(tag_dev, pred) #对比分类预测结果和人工标注的正确结果，给出分类器准确度
'''
以下为对数据进行格式化存储
'''
# with open('./weibofile/weibo-seg-0.txt') as f:
#     pos_review=[]
#     for line in f:
#         wrodlist=[]
#         recs=line.split()
#         for word in recs:
#             wrodlist.append(word)
#         pos_review.append(wrodlist)
# pickle.dump(pos_review, open("./weibofile"+'/'+'pos_review'+'.pkl','wb'))
# with open('./weibofile/weibo-seg-1.txt') as f:
#     neg_review=[]
#     for line in f:
#         wrodlist=[]
#         recs=line.split()
#         for word in recs:
#             wrodlist.append(word)
#         neg_review.append(wrodlist)
# pickle.dump(neg_review, open("./weibofile"+'/'+'neg_review1'+'.pkl','wb'))
pos_review = pickle.load(open('weibofile/pos_review.pkl','r'))
neg_review = pickle.load(open('weibofile/neg_review1.pkl','r'))
shuffle(pos_review) #把积极文本的排列随机化
shuffle(neg_review)
# size = int(len(pos_review)/2 - 18)
# pos = pos_review[:size]
pos = pos_review
neg = neg_review
'''
使用所有词作为特征
BernoulliNB`s accuracy is 0.640000
MultinomiaNB`s accuracy is 0.640000
LogisticRegression`s accuracy is 0.620000
SVC`s accuracy is 0.560000
LinearSVC`s accuracy is 0.620000
NuSVC`s accuracy is 0.630000
'''
# posFeatures = feature.pos_features(feature.bag_of_words,pos) #使用所有词作为特征
# negFeatures = feature.neg_features(feature.bag_of_words,neg)
'''
使用双词搭配作特征
BernoulliNB`s accuracy is 0.610000
MultinomiaNB`s accuracy is 0.610000
LogisticRegression`s accuracy is 0.590000
SVC`s accuracy is 0.510000
LinearSVC`s accuracy is 0.570000
NuSVC`s accuracy is 0.550000
'''
# posFeatures = feature.pos_features(feature.bigram,pos)
# negFeatures = feature.neg_features(feature.bigram,neg)

'''
使用所有词加上双词搭配作特征
BernoulliNB`s accuracy is 0.700000
MultinomiaNB`s accuracy is 0.710000
LogisticRegression`s accuracy is 0.720000
SVC`s accuracy is 0.690000
LinearSVC`s accuracy is 0.660000
NuSVC`s accuracy is 0.710000
'''
# posFeatures = feature.pos_features(feature.bigram_words,pos)
# negFeatures = feature.neg_features(feature.bigram_words,neg)

'''
使用卡方统计量（Chi-square）来选择信息量丰富的特征(以所有词作为特征)
BernoulliNB`s accuracy is 0.810000
MultinomiaNB`s accuracy is 0.800000
LogisticRegression`s accuracy is 0.750000
SVC`s accuracy is 0.640000
LinearSVC`s accuracy is 0.770000
NuSVC`s accuracy is 0.740000

'''
# posFeatures = pos_features(feature_extraction_method1,pos)
# negFeatures = neg_features(feature_extraction_method1,neg)

'''
使用卡方统计量（Chi-square）来选择信息量丰富的特征（以所有词加双词作为特征）
耗时太长暂未实验
'''
posFeatures = pos_features(feature_extraction_method2,pos)
negFeatures = neg_features(feature_extraction_method2,neg)
train = posFeatures[174:]+negFeatures[174:]
devtest = posFeatures[124:174]+negFeatures[124:174]
test = posFeatures[:124]+negFeatures[:124]
print 'BernoulliNB`s accuracy is %f' %score(BernoulliNB())
print 'MultinomiaNB`s accuracy is %f' %score(MultinomialNB())
print 'LogisticRegression`s accuracy is %f' %score(LogisticRegression())
print 'SVC`s accuracy is %f' %score(SVC())
print 'LinearSVC`s accuracy is %f' %score(LinearSVC())
print 'NuSVC`s accuracy is %f' %score(NuSVC())

'''
要对新的数据进行分类时，现将数据处理为指定格式后读入
def extract_features(data):
    feat = []
    for i in data:
        feat.append(feature_extraction_method2(i))
    return feat
moto_features = extract_features(moto) #把文本转化为特征表示的形式
然后载入训练好的分类器clf，即可进行分类
pred = clf.batch_prob_classify(moto_features) #该方法是计算分类概率值的
'''