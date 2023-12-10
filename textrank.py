import pandas as pd
import jieba.analyse
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
"""
   TextRank权重：
        1、将待抽取关键词的文本进行分词、去停用词、筛选词性
        2、以固定窗口大小(默认为5，通过span属性调整)，词之间的共现关系，构建图
        3、计算图中节点的PageRank，注意是无向带权图
"""

def getKeywords_textrank(data,topK):
    idList,titleList,abstractList = data['id'],data['title'],data['abstract']
    for index in range(len(idList)):
        text = '%s。%s' % (titleList[index], abstractList[index]) # 拼接标题和摘要   拼接数据
        jieba.analyse.set_stop_words("./resource/stopWord.txt") # 加载自定义停用词表
        #text为每一篇文章               allowPOS：返回符合要求词性的关键词
        keywords = jieba.analyse.textrank(text, topK=topK, allowPOS=('n','nz','v','vd','vn','l','a','d',"x"))  # TextRank关键词提取，词性筛选
        print(text)
        print(keywords)
        print("\n")

if __name__ == '__main__':
    dataFile = './resource/sample_data.csv'
    data = pd.read_csv(dataFile)  # 加载数据

    result = getKeywords_textrank(data, 10)