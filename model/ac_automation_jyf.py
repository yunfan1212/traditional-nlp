#coding=utf-8

__all__=["Ahocorasics",]
from sklearn.externals import joblib

class Node(object):
    def __init__(self):
        self.next={}
        self.fail=None
        self.isWord=False

class Ahocorasics(object):
    def __init__(self):
        self.__root=Node()
        self.node={"root":{}}

    def addWord(self,word):
        '''字典树的创建，add words to tree'''
        father=self.__root
        for i in range(0,len(word)):
            if word[i] not in father.next:    #无该字的孩子节点
                father.next[word[i]]=Node()   #创建孩子节点
            father=father.next[word[i]]       #向下遍历
        father.isWord=True
        return

    def make(self):
        '''创建自动机，构造失效函数，BFS广度遍历方法'''
        fatherQueue=[]
        fatherQueue.append(self.__root)
        while len(fatherQueue)>0:
            father=fatherQueue.pop()        #右侧， 父节点
            for child,child_child_node in father.next.items():
                if father==self.__root:
                    father.next[child].fail=self.__root
                else:
                    p=father.fail
                    while p is not None:
                        if child in p.next:
                            father.next[child].fail=p.next[child]
                            break
                        p=p.fail
                    if p is None:
                        father.next[child].fail=self.__root
                fatherQueue.append(father.next[child])

    def ac_save(self):
        result=[]
        root=self.__root
        queue=[root]
        while len(queue)>0:
            father=queue.pop(0)
            result.append((father.next,father.fail,father.isWord))

            for k,value in father.next.items():
                if k is not None:
                    queue.append(value)
        import numpy as np
        joblib.dump(root, 'lr.model')
        lr = joblib.load('lr.model')


    def ac_load(self):
        self.__root=joblib.load("lr.model")
        return

    def delete_string(self,words):
        pass


    def search(self, content):
        '''查找字符串'''
        father=self.__root
        result,result1=[],[]
        startWordIndex=0
        currentPosition=0
        while currentPosition<len(content):
            word=content[currentPosition]
            while father!=None and word not in father.next and father!=self.__root:
                father=father.fail
            if father!=None and word in father.next:
                if father==self.__root:
                    startWordIndex=currentPosition
                father=father.next[word]
            else:
                father=self.__root
            if father.isWord:
                result.append((startWordIndex,currentPosition))
                result1.append(content[startWordIndex:currentPosition+1])
            currentPosition+=1
        return result,result1


if __name__ == '__main__':
    ah = Ahocorasics()
    ah.addWord('E-MAIL')

    ah.addWord("我是")
    ah.addWord("贾亚飞")

    ah.make()
    x, y = ah.search('E-MAIL123我是好人，测试贾亚飞')
    print(x)
    print(y)
    ah.ac_save()
    ahmodel=Ahocorasics()
    ahmodel.ac_load()
    x,y=ahmodel.search('E-MAIL123我是好人，测试贾亚飞')
    print(x)
    print(y)