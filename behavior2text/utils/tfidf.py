import requests
def tfidf(apiDomain, context):
    tfidf = requests.post(apiDomain + '/tfidf/tfidf?flag=n', data={'doc':context}).json()
    return [(key, {'key':{key:1}, 'count':value}) for key, value in tfidf]