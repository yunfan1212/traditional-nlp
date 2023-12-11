import numpy as np
from scipy.special import gammaln
import re
import matplotlib.pyplot as plt


class LDACGS:
    """Do LDA with Gibbs Sampling."""
    def __init__(self, n_topics, alpha=0.1, beta=0.1):
        """参数初始化"""
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta

    def build_corpus(self, filename, stopwords_file=None):
        """阅读文本，并创建词典"""
        with open(filename, 'r',encoding="utf8") as infile:
            doclines = [re.sub(r'[^\w ]', '', line.lower()).split(' ') for line in infile]   #读取文档，并分词
        #创建词典
        n_docs = len(doclines)            #文档数
        self.vocab = list({v for doc in doclines for v in doc})       #字典集合
        if stopwords_file:
            with open(stopwords_file, 'r') as stopfile:
                stops = stopfile.read().split()
            self.vocab = [x for x in self.vocab if x not in stops]   #去停用词
        self.vocab.sort()   #字典排序
        #将文本转为id形式
        self.documents = []
        for i in range(n_docs):
            self.documents.append({})         #{词下标: 词id,}
            for j in range(len(doclines[i])):
                if doclines[i][j] in self.vocab:
                    self.documents[i][j] = self.vocab.index(doclines[i][j])     #将文本转为id，并不等长

    def initialize(self):
        """参数初始化"""
        self.n_words = len(self.vocab)                            #单词数量
        self.n_docs = len(self.documents)                         #文本数量

        self.nmz = np.zeros((self.n_docs, self.n_topics))         #文档-话题矩阵

        self.nzw = np.zeros((self.n_topics, self.n_words))        #话题-单词矩阵

        self.nz = np.zeros(self.n_topics)                         #话题数量

        self.topics = {}  # key-value pairs of form (m,i):z  #主题词典

        for m in range(self.n_docs):
            for i in self.documents[m]:                     #文档中的每个词

                z = np.random.choice(self.n_topics)        #随机选择一个主题
                # Retrieve vocab index for i-th word in document m.
                w = self.documents[m][i]                  #获取词汇id
                # Increment count matrices
                self.nmz[m][z] += 1                     #文档-话题矩阵 值+1
                self.nzw[z][w] += 1                     #话题-单词矩阵  值+1
                self.nz[z] += 1                         #话题对应词汇数+1
                # Store topic assignment, i.e. self.topics[(m,i)]=z
                self.topics[(m, i)] = z         #为每个词分配主题

    def sample(self, filename, burnin=100, sample_rate=10, n_samples=10, stopwords=None):
        self.build_corpus(filename, stopwords)                      #创建词典
        self.initialize()                                               #参数初始化

        self.total_nzw = np.zeros((self.n_topics, self.n_words))         #话题-单词 矩阵
        self.total_nmz = np.zeros((self.n_docs, self.n_topics))          #文本-主题矩阵
        self.logprobs = np.zeros(burnin + sample_rate * n_samples)       #概率日志

        # Problem 5:
        for i in range(burnin):         #轮次
            # Sweep and store log likelihood.
            self._sweep()   #训练
            self.logprobs[i] = self._loglikelihood()             #存储损失函数

        for i in range(n_samples * sample_rate):                 #采样优化
            # Sweep and store log likelihood
            self._sweep()
            self.logprobs[burnin + i] = self._loglikelihood()

            if not i % sample_rate:
                # accumulate counts
                self.total_nmz += self.nmz
                self.total_nzw += self.nzw
        '''
        一般在收敛之前，需要跑一定数量的采样次数让采样程序大致收敛，这个次数一般称为：burnin period。
        我们希望从采样程序中获得独立的样本，但是显然以上的采样过程明显依赖上一个采样结果，那么我们可以在上一次采样后，
        先抛弃一定数量的采样结果，再获得一个采样结果，这样可以大致做到样本独立，这样真正采样的时候，总有一定的滞后次数，
        这样的样本与样本的间隔称为：SampleLag。
        '''



    def phi(self):            #求模型参数fai
        phi = self.total_nzw + self.beta
        self._phi = phi / np.sum(phi, axis=1)[:, np.newaxis]

    def theta(self):         #求模型参数theta
        theta = self.total_nmz + self.alpha
        self._theta = theta / np.sum(theta, axis=1)[:, np.newaxis]

    def topterms(self, n_terms=10):          #预测前n_terms个单词
        self.phi()
        self.theta()     #获取训练参数
        vec = np.atleast_2d(np.arange(0, self.n_words))        #将数组转为2维
        topics = []
        for k in range(self.n_topics):      #遍历话题
            probs = np.atleast_2d(self._phi[k, :])          #话题-单词分布   k 话题下 单词的概率分布
            mat = np.append(probs, vec, 0)         #在0 维度上将词汇id 拼接到概率上，使之一一对应
            sind = np.array([mat[:, i] for i in np.argsort(mat[0])]).T     #按概率大小，从大到小排序
            topics.append([self.vocab[int(sind[1, self.n_words - 1 - i])] for i in range(n_terms)])    #获取n_terms个词汇
        return topics

    def toplines(self, n_lines=5):                  #求主题句
        lines = np.zeros((self.n_topics, n_lines))
        for i in range(self.n_topics):
            args = np.argsort(self._theta[:, i]).tolist()
            args.reverse()
            lines[i, :] = np.array(args)[0:n_lines] + 1
        return lines

    def _remove_stopwords(self, stopwords):
        return [x for x in self.vocab if x not in stopwords]

    def _conditional(self, m, w):
        dist = (self.nmz[m, :] + self.alpha) * (self.nzw[:, w] + self.beta) / (self.nz + self.beta * self.n_words)
        return dist / np.sum(dist)

    def _sweep(self):
        for m in range(self.n_docs):             #迭代文档
            for i in self.documents[m]:
                # Problem 4:
                # Retrieve vocab index for i-th word in document m.
                w = self.documents[m][i]    #获取单词id
                # Retrieve topic assignment for i-th word in document m.
                z = self.topics[(m, i)]      #获取对应主题
                # Decrement count matrices.
                self.nmz[m][z] -= 1           #文档-主题 矩阵-1
                self.nzw[z][w] -= 1            #主题-单词矩阵-1
                self.nz[z] -= 1                #主题对应词汇数-1
                # Get conditional distribution.
                dist = self._conditional(m, w)        #对话题进行重新抽样
                # Sample new topic assignment.
                new_z = np.random.choice(self.n_topics, p=dist)    #重新选择话题
                # Increment count matrices.
                self.nmz[m][new_z] += 1               #文档-话题+1
                self.nzw[new_z][w] += 1               #话题-单词+1
                self.nz[new_z] += 1
                # Store new topic assignment.
                self.topics[(m, i)] = new_z          #存储新主题

    def _loglikelihood(self):   #目的查看是否收敛,  log似然函数
        lik = 0                #self.nzw 话题-单词矩阵   gammaln: gamma函数绝对值的对数  ln(|gamma(x)|)
        for z in range(self.n_topics):
            lik += np.sum(gammaln(self.nzw[z, :] + self.beta)) - gammaln(np.sum(self.nzw[z, :] + self.beta))
            lik -= self.n_words * gammaln(self.beta) - gammaln(self.n_words * self.beta)
            for m in range(self.n_docs):       #nmz 文档-话题
                lik += np.sum(gammaln(self.nmz[m, :] + self.alpha)) - gammaln(np.sum(self.nmz[m, :] + self.alpha))
                lik -= self.n_topics * gammaln(self.alpha) - gammaln(self.n_topics * self.alpha)
        return lik


def plot_logprobs(logprobs):
    # Problem 6:
    plt.plot(logprobs)
    plt.show()


def main():
    fpath = "./data/reagan.txt"
    swpath = "./data/stopwords_en.txt"
    lda = LDACGS(10)
    lda.sample(fpath, stopwords=swpath, burnin=20)
    for i, topic in enumerate(lda.topterms()):
        print(i, topic)
    print("========================")
    for i, line in enumerate(lda.toplines()):
        print(i, line)

    plot_logprobs(lda.logprobs)


if __name__ == '__main__':
    np.random.seed(1)
    main()
