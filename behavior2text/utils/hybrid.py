import requests
from .kcemCluster import kcemCluster

def hybrid(apiDomain, wordCount, context):
    tfidfDict = dict(requests.post(apiDomain + '/tfidf/tfidf?flag=n', data={'doc':context}).json())
    result = kcemCluster(apiDomain, wordCount)

    refinedResult = []
    for hypernym, dictionary in result:
        delList = []
        for term, tf in dictionary['key'].items():
            value = tfidfDict.get(term, 0)
            if not value:
                delList.append(term)
            else:
                dictionary['key'][term] = value
        for term in delList:
            del dictionary['key'][term]

        if dictionary['key']:
            refinedResult.append((hypernym, dictionary))
    return sorted(refinedResult, key=lambda x:(-x[1]['count'], -max(x[1]['key'].values(), key=lambda y:y, default=0)))
