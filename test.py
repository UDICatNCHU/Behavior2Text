from __init__ import Behavior2Text
import sys, pyprind, subprocess

modeList = ['tfidf', 'kcem', 'kcemCluster', 'hybrid', 'contextNetwork']

for mode in modeList:
    b = Behavior2Text(mode)
    b.buildTopn()
    topnsFile = json.load(open(b.output, 'r'))
    for refineData, fileName in topnsFile:
        topn = b.getTopN(refineData, b.topNum) if refineData else []

        if not topn:
            print([])
        else:
            print(b.sentence(topn, fileName))
    print('NDCG is {}'.format(b.NDCG/len(topnsFile)))