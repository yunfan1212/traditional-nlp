
def KMP_algorithm(string, substring):
  '''
  KMP字符串匹配的主函数
  若存在字串返回字串在字符串中开始的位置下标，或者返回-1
  '''
  pnext = gen_pnext(substring)
  n = len(string)
  m = len(substring)
  i, j = 0, 0
  while (i<n) and (j<m):
    if (string[i]==substring[j]):
      i += 1
      j += 1
    elif (j!=0):
      j = pnext[j-1]
    else:
      i += 1
  if (j == m):
    return [i-j,i]
  else:
    return -1,-1

def gen_pnext(substring):
  """
  构造临时数组pnext
  """
  index, m = 0, len(substring)
  pnext = [0]*m
  i = 1
  while i < m:
    if (substring[i] == substring[index]):
      pnext[i] = index + 1
      index += 1
      i += 1
    elif (index!=0):
      index = pnext[index-1]
    else:
      pnext[i] = 0
      i += 1
  return pnext
if __name__ == "__main__":
  string = '我和妈妈去吃饭，想找一家有酸辣蕨根粉的餐馆，最好妈妈人均消费在100-150元之间的，评分在4分以上的，有推荐的吗？'
  substring = '妈妈'
  out = KMP_algorithm(string, substring)
  print(out)