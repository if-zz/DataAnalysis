# -*- coding:utf-8 -*-
import os
import string
import sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
reload(sys)
sys.setdefaultencoding('utf8')
#------------------------此部分是将分词结果进行TFIDF处理，得出在不同条目中，各词语的权值，并按条目存储
def Tfidf(filelist):
    corpus=[]
    for ff in filelist:
        fname = ff
        f = open(fname, 'r+')
        content = f.read()
        f.close()
        corpus.append(content)
    vectorizer=CountVectorizer()
    transformer=TfidfTransformer()
    tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))
    word=vectorizer.get_feature_names()#所有文本关键字
    weight=tfidf.toarray() #对应的tfidf矩阵
    sFilePath = './tfidffile'
    if not os.path.exists(sFilePath):
        os.mkdir(sFilePath)
    for i in range(
        len(weight)):
        print u"--------Writing all the tf-idf in the", i, u" file into ",sFilePath+'/'+ string.zfill(i+1, 5) + '.txt', "--------"
        f = open(sFilePath+'/'+ string.zfill(i+1, 5) + '.txt', 'w+')
        for j in range(len(word)):
            f.write(word[j] + "	" + str(weight[i][j]) + "\n")
        f.close()

filelist=[]
files=os.listdir("segfile")
for f in files :
    if f[0]== '.':
        pass
    else:
        f = "segfile/" + f
        filelist.append(f)
Tfidf(filelist)