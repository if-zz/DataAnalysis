# -*- coding:utf-8 -*-
import jieba
import sys
reload(sys)
sys.setdefaultencoding('utf8')
# ------------------------此部分为将爬取的数据进行分词，并将结果按照每条内容分别存储
Data=[]
# 处理爬取的数据
# with open('G:\\PyCharm 2016.2.3\\DCSpider\\items.txt') as f:
#     for line in f:
#         recs = line.split()
#         data = recs[2]
#         Data.append(data)

#处理已分类的微博数据，用来进行机器学习，由于条件限制，现只读取前1500条内容
# 0：喜悦
# 1：愤怒
# 2：厌恶
# 3：低落
lines=0
with open('0_simplifyweibo.txt') as f:
    for line in f:
       Data.append(line)
       lines=lines+1
       if lines>1500:
           break
seg_list=[]
stopwords = [line.strip().decode('utf-8') for line in open('stopwords.txt').readlines()]
for data in Data:
    data=jieba.cut(data)
    # seg_list.append(data)
    # print("**".join(data))
    cut_data=[]
    for word in data:
        if word not in stopwords:
           cut_data.append(word)
    seg_list.append(cut_data)
i = 1
f = open("./weibofile/" + "weibo-seg-0.txt", 'w')
for seg in seg_list:
    result_word=[]
    for word in seg:
        word = ''.join(word.split())
        if (word != '' and word != "\n" and word != "\n\n"):
            result_word.append(word)
    f.write((' '.join(result_word))+'\n')
f.close()

