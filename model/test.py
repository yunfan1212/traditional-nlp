import numpy as np
from sklearn.externals import joblib


class Node(object):
    def __init__(self):
        self.next={}
        self.fail=None
        self.isWord=False

lr = joblib.load('lr.model')
print(lr.next)