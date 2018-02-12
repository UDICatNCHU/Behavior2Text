from __init__ import Behavior2Text
import sys, pyprind, subprocess, json, os

modeList = ['tfidf', 'kcem', 'kcemCluster', 'hybrid', 'contextNetwork', 'pagerank']

for topn in range(3, 10, 3):
    for clusterTopn in range(3, 10, 3):
        print("topn: {}, clusterTopn:{}".format(topn, clusterTopn))
        print('======================================')
        for mode in modeList:
            b = Behavior2Text(mode, topn, clusterTopn)
            b.buildTopn()
            topnsFile = json.load(open(b.output, 'r'))
            for refineData, fileName in topnsFile:
                topn = b.getTopN(refineData, b.topNum) if refineData else []

                if not topn:
                    print([])
                else:
                    print(b.sentence(topn, fileName))
            print('NDCG is {}'.format(b.NDCG/len(topnsFile)))
            os.remove(mode + '.json')