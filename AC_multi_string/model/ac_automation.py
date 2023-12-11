
__all__ = ['Ahocorasick', ]

#多字符串模式匹配
class Node(object):
    def __init__(self):
        self.next = {}
        self.fail = None
        self.isWord = False


class Ahocorasick(object):
    def __init__(self):
        self.__root = Node()

    def addWord(self, word):
        '''
            @param word: add word to Tire tree
                            添加关键词到Tire树中
        '''
        tmp = self.__root
        for i in range(0, len(word)):
            if word[i] not in tmp.next:
                tmp.next[word[i]] = Node()
            tmp = tmp.next[word[i]]
        tmp.isWord = True                #代表叶子节点

    def make(self):
        '''
            build the fail function
            构建自动机，失效函数
        '''
        tmpQueue = []
        tmpQueue.append(self.__root)       #Bfs 遍历字典树
        while (len(tmpQueue) > 0):
            temp = tmpQueue.pop()         #BFS遍历，temp父节点
            for k, v in temp.next.items():      #temp为父节点,k为孩子节点，v为孩子节点的子节点
                if temp == self.__root:          #根节点的孩子们fail指向根节点，因为root失效指针是本身
                    temp.next[k].fail = self.__root
                else:                             #非根节点
                    p = temp.fail                 #父节点的失效指针 指向
                    while p is not None:           #父节点的失效指针不为None
                        if k in p.next:            # 孩子节点=父节点 失效指针 节点的子节点
                            temp.next[k].fail = p.next[k]     #第k个孩子节点指向父节点的子节点
                            break
                        p = p.fail                       #未找到，则向上一层找失配指针
                    if p is None:                  #如果失效指针指向root，则孩子节点指向根节点
                        temp.next[k].fail = self.__root
                tmpQueue.append(temp.next[k])      #同层节点添加到队列中

    def search(self, content):
        '''
            @return: a list of tuple,the tuple contain the match start and end index
        '''
        p = self.__root
        result = []
        result1=[]
        startWordIndex = 0
        endWordIndex = -1
        currentPosition = 0

        while currentPosition < len(content):
            word = content[currentPosition]
            # 检索状态机，直到匹配
            print(p.next)
            while p!=None and word not in p.next and p != self.__root:
                p = p.fail      #匹配不到，且不是根节点，查找失效指针

            if p!=None and word in p.next:     #匹配到
                if p == self.__root:
                    # 若当前节点是根且存在转移状态，则说明是匹配词的开头，记录词的起始位置
                    startWordIndex = currentPosition
                # 转移状态机的状态
                p = p.next[word]
            else:                  #找不到最大前缀，从root开始查找新下词
                p = self.__root

            if p.isWord:      #找到叶子节点，则该词找到
                # 若状态为词的结尾，则把词放进结果集
                result.append((startWordIndex, currentPosition))
                result1.append(content[startWordIndex:currentPosition+1])
            currentPosition += 1
        return result,result1




if __name__ == '__main__':
    ah = Ahocorasick()
    ah.addWord('测试')
    ah.addWord("我是")
    ah.addWord("贾亚飞")
    ah.make()
    x,y=ah.search('测试123我是好人，测试贾亚飞')

    z=ah.replace('测试123我是好人，测试贾亚飞')
