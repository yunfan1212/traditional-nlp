#coding=utf8

'''jaccard 模型的实现，计算字符串的相似度=字符的交集/字符的并集
 此处采用工具计算交集与并集
'''

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

import Levenshtein

def jaccard_similarity(s1, s2):
    def add_space(s):
        return ' '.join(list(s))

    # 将字中间加入空格
    s1, s2 = add_space(s1), add_space(s2)

    # 转化为TF矩阵
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()

    # 求交集
    numerator = np.sum(np.min(vectors, axis=0))
    # 求并集
    denominator = np.sum(np.max(vectors, axis=0))
    # 计算杰卡德系数
    return 1.0 * numerator / denominator




if __name__ == '__main__':
    s1 = '你在干嘛呢'
    s2 = '你在干什么呢'
    print(jaccard_similarity(s1, s2))
    y = Levenshtein.jaro_winkler(s1, s2)
    print(y)











