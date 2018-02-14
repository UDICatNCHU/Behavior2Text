import numpy as np
import requests
def doc2vec(apiDomain, wordCount):
    vec = np.zeros(400)
    for word, count in wordCount.items():
        vec += np.array(requests.get(apiDomain + '/kem/vector?keyword={}'.format(word)).json()['value']) * count
    return vec / sum(wordCount.values())