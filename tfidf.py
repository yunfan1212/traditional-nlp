from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer       #特征获取模块
import jieba.posseg
import pandas as pd


"""
       TF-IDF权重：
           1、CountVectorizer 构建词频矩阵
           2、TfidfTransformer 构建tfidf权值计算
           3、文本的关键字
           4、对应的tfidf矩阵
"""
# 数据预处理操作：分词，去停用词，词性筛选
def dataPrepos(text, stopkey):
    l = []
    pos = ['n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd']  # 定义选取的词性
    seg = jieba.posseg.cut(text)  # 分词
    for i in seg:
        if i.word not in stopkey and i.flag in pos:  # 去停用词 + 词性筛选
            l.append(i.word)
    return l

# tf-idf获取文本top10关键词
def getKeywords_tfidf(data,stopkey):

    idList, titleList, abstractList = data['id'], data['title'], data['abstract']
    corpus = [] # 将所有文档输出到一个list中，一行就是一个文档
    for index in range(len(idList)):
        text = '%s。%s' % (titleList[index], abstractList[index]) # 拼接标题和摘要
        text = dataPrepos(text,stopkey) # 文本预处理
        text = " ".join(text) # 连接成字符串，空格分隔
        corpus.append(text)

    # 1、构建词频矩阵，将文本中的词语转换成词频矩阵
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus) # 词频矩阵,a[i][j]:表示j词在第i个文本中的词频
    # 2、获取词袋模型中的关键词
    word = vectorizer.get_feature_names()    #获取每个词

    # 3、统计每个词的tf-idf权值
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    # 4、获取tf-idf矩阵，a[i][j]表示j词在i篇文本中的tf-idf权重
    weight = tfidf.toarray()

    # 5、打印词语权重
    ids, titles, keys = [], [], []
    for i in range(len(weight)):
        print (u"-------这里输出第", i+1 , u"篇文本的词语tf-idf------")
        ids.append(idList[i])
        titles.append(titleList[i])
        df_word,df_weight = [],[] # 当前文章的所有词汇列表、词汇对应权重列表
        for j in range(len(word)):
            print (word[j],weight[i][j])
            df_word.append(word[j])
            df_weight.append(weight[i][j])



def main():
    # 读取数据集
    dataFile = './resource/sample_data.csv'
    data = pd.read_csv(dataFile)
    # 停用词表
    stopkey = [w.strip() for w in open('./resource/stopWord.txt', 'r',encoding="utf-8").readlines()]
    # tf-idf关键词抽取
    getKeywords_tfidf(data,stopkey)


if __name__ == '__main__':
    main()