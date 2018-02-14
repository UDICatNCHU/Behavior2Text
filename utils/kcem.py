import requests
from .doc2vec import doc2vec
from collections import defaultdict

def kcem(apiDomain, wordCount):
    docVec = doc2vec(apiDomain, wordCount)
    kcemList = requests.get(apiDomain + '/kcem/kcemList?keywords={}'.format('+'.join(wordCount))).json()
    ###doc2vec version####
    # result = requests.post(self.apiDomain + '/kcem?keyword={}'.format('+'.join(wordCount)), data={'counter':json.dumps(wordCount)}).json()
    ######################
    def countHypernym(kcemList):
        result = defaultdict(dict)
        for kcem in kcemList:
            if not kcem['value']:
                continue
            hypernym = kcem['value'][0][0]
            ngramKey = kcem['key']
            originKey = kcem['origin']
            termFrequency = wordCount[originKey]

            # 把hypernym原始的查詢key給紀錄起來，他在文本中出現幾次也是
            result[hypernym]['key'][ngramKey] = result[hypernym].setdefault('key', {}).setdefault(ngramKey, 0) + termFrequency
            result[hypernym]['count'] = result[hypernym].setdefault('count', 0) + termFrequency
        return result
    result = countHypernym(kcemList)
    return sorted(result.items(), key=lambda x:-x[1]['count'])
