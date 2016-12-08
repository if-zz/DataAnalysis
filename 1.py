# -*- coding:utf-8 -*-
import jieba
import sys
reload(sys)
sys.setdefaultencoding('utf8')
# ------------------------此部分为将爬取的数据进行分词，并将结果按照每条内容分别存储
Data=[]
# 处理爬取的数据
with open('G:\\PyCharm 2016.2.3\\DCSpider\\items.txt') as f:
    for line in f:
        recs = line.split()
        data = recs[2]
        Data.append(data)
lines=0
# with open('2_simplifyweibo.txt') as f:
#     for line in f:
#        Data.append(line)
#        lines=lines+1
#        if lines>1500:
#            break
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
f = open("./segfile/" + "dacheng-seg.txt", 'w')
for seg in seg_list:
    result_word=[]
    for word in seg:
        word = ''.join(word.split())
        if (word != '' and word != "\n" and word != "\n\n"):
            result_word.append(word)
    f.write((' '.join(result_word))+'\n')
f.close()

