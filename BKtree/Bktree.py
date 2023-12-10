
#快速计算 字符串间的相似度
#字符串的模糊匹配，可用于召回，计算两个字符串的模糊匹配

import os
from Levenshtein import distance

class Node(object):
    def __init__(self,word):
        self.word=word
        self.children={}
    def __repr__(self):
        return "<Node: %r>"%self.word



class BKTree():
    def __init__(self,diction,distance_func=distance):
        self.root=None
        self.dist_func=distance_func
        self.diction=self.load_diction(diction)

    '''加载词汇'''
    @staticmethod
    def load_diction(diction):
        diction=os.path.join(os.path.dirname(os.path.abspath(__file__)),diction)
        with open(diction,"r",encoding="utf-8") as f:
            lines=f.readlines()
            lines=[l.strip() for l in lines if len(l.strip())>0]
        return set(lines)

    '''构建BKtree'''
    def build_tree(self):
        for w in self.diction:
            self.add(w)
    def add(self,word):
        if self.root is None:
            self.root=Node(word)
            return
        node=Node(word)
        cur=self.root

        dist=self.dist_func(word,cur.word)
        while dist in cur.children:
            cur=cur.children[dist]
            dist=self.dist_func(word,cur.word)
        cur.children[dist]=node
        node.parent=cur

    '''查询'''
    def search(self,word,max_dist):
        candidate=[self.root]
        found=[]
        while len(candidate)>0:
            node=candidate.pop(0)
            dist=self.dist_func(word,node.word)
            if dist<=max_dist:
                found.append(node)
            for child_dist,child in node.children.items():
                if dist-max_dist<=child_dist<=dist+max_dist:
                    candidate.append(child)

        if found:
            found=[f.word for f in found]
        return found

if __name__ == '__main__':
    tree = BKTree('./kwds_credit_report.txt')
    tree.build_tree()
    print(tree.search('以还笨金', 2))




















