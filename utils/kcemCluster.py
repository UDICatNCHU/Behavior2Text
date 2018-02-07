import requests
from .doc2vec import doc2vec

def kcemCluster(apiDomain, wordCount):
    def simpleUnion(clusters, unionList):
        finalCluster = clusters[unionList[0]]
        for cluster in clusters:
            if cluster['groupIdx'] in unionList:
                finalCluster['key'].update(cluster['key'])
                finalCluster['Allhypernym'].update(cluster['Allhypernym'])
        clusters = [cluster for cluster in clusters if cluster['groupIdx'] not in unionList[0:]] + [finalCluster]
        for groupIdx, cluster in enumerate(clusters):
            cluster['groupIdx'] = groupIdx
        return clusters

    def clustering(kcemList):
        clusters = []
        for kcem in kcemList:
            # kcem最頂端的keyword是沒有hypernym的，其他太爛的字也沒有
            if not kcem['value']:
                continue

            # union的條件是keyword自身以及hypernym有intersection就union
            originKcemKey = kcem['origin']
            termFrequency = wordCount[originKcemKey]
            kcemDict = dict({originKcemKey:termFrequency}, **{hypernym:termFrequency for hypernym, possibility in kcem['value'] if hypernym != '包含規範控制信息的維基百科條目'})

            insert = False
            unionList = []
            for cluster in clusters:
                groupIdx = cluster['groupIdx']

                # 除了keyword自身的hypernyms以外，keyword自身也包含在內，只要有overlap就歸在同一群
                intersection = set(cluster['Allhypernym']).intersection(set(kcemDict))
                if intersection:
                    insert = True
                    unionList.append(groupIdx)

                    cluster['key'].update({originKcemKey:termFrequency})
                    for hypernym in kcemDict:
                        cluster['Allhypernym'][hypernym] = cluster['Allhypernym'].setdefault(hypernym, 0) + termFrequency

            if not insert:
                # 要注意，因為kcem有套用ngram搜尋，所以kcem的key是可能會重複的喔!!!
                clusters.append({'key':{originKcemKey:termFrequency}, 'Allhypernym':kcemDict.copy(), 'groupIdx':len(clusters)})
            else:
                # 如果有insert過，就要判斷是否需要union
                if len(unionList) >= 2:
                    clusters = simpleUnion(clusters, unionList)

        return clusters

    docVec = doc2vec(apiDomain, wordCount)
    kcemList = requests.get(apiDomain + '/kcem/kcemList?keywords={}'.format('+'.join(wordCount))).json()
    # sorting and format
    result = {}
    for cluster in clustering(kcemList):
        cluster['hypernym'] = sorted(cluster['Allhypernym'].items(), key=lambda x:-x[1])[0][0]
        result[cluster['hypernym']] = {'key':{k:wordCount[k] for k in cluster['key']}}
        result[cluster['hypernym']]['count'] = sum([wordCount[k] for k in cluster['key']])
    return sorted(result.items(), key=lambda x:-x[1]['count'])