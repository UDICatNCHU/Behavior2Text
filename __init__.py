import requests, os, json, threading
from collections import defaultdict, Counter
from scipy import spatial
import numpy as np
from itertools import takewhile
from udicOpenData.stopwords import rmsw


class Behavior2Text(object):
    def __init__(self, mode):
        self.template = json.load(open('template.json', 'r'))
        self.topNum = 3
        self.accessibility_log = 'goodHuman'
        # self.accessibility_log = 'test'
        self.mode = mode
        self.output = '{}.json'.format(self.mode)
        self.EntityOnly = False
        # self.EntityOnly = True

        self.DEBUG = True
        if self.DEBUG:
            self.apiDomain = 'http://140.120.13.243'
        else:
            self.apiDomain = 'http://udiclab.cs.nchu.edu.tw/'

    @staticmethod
    def getTopN(topList, n):
        n = n if n < len(topList) else -1
        # minCount = topList[n][1]['count']
        # return list(takewhile(lambda x:x[1]['count'] >= minCount, topList))
        return topList[:n]

    def sentence(self, topn):
        def selectBestTemplate():
            def calTemplateSim(TemplateCandidate, templateIndex, template):
                for replaceIndex in template['replaceIndices']:
                    replaceWord = template['key'][replaceIndex]
                    TemplateCandidate[templateIndex].setdefault(replaceWord, {})

                    for _, topnKeywordDict in topn:
                        topnKeyword = sorted(topnKeywordDict['key'].items(), key=lambda x:-x[1])[0][0]
                        similarity = requests.get(self.apiDomain + '/kem/similarity?k1={}&k2={}'.format(replaceWord, topnKeyword)).json()
                        if similarity == {}:
                            continue
                        TemplateCandidate[templateIndex][replaceWord][topnKeyword] = similarity['similarity']
                    TemplateCandidate[templateIndex]['sum'] = TemplateCandidate[templateIndex].setdefault('sum', 0) + max(TemplateCandidate[templateIndex][replaceWord].values(), default=0) / len(template['replaceIndices'])

            TemplateCandidate = defaultdict(dict)
            for templateIndex, template in enumerate(self.template):
                calTemplateSim(TemplateCandidate, templateIndex, template)

            # select most possible template to generate sentence
            index, templateKeywords = sorted(TemplateCandidate.items(), key=lambda x:-x[1]['sum'])[0]
            del templateKeywords['sum']
            return index, templateKeywords

        def generate(index, templateKeywords, raw=False):
            select = set()
            result = {}
            for templateKeyword, templateKeywordCandidates in sorted(templateKeywords.items(), key=lambda x:max(x[1].items(), key=lambda x:x[1])[1], reverse=True):
                # 最好的情況是填入template的詞不要重複
                # 但是真的沒辦法也只能重複填了...
                candidate = [(templateKeywordCandidate, similarity) for templateKeywordCandidate, similarity in templateKeywordCandidates.items() if templateKeywordCandidate not in select]
                if candidate == []:
                    candidate = [(templateKeywordCandidate, similarity) for templateKeywordCandidate, similarity in templateKeywordCandidates.items()]

                # use multiple key for max
                # Because in some case, candidate has same similarity
                # so need to use key length to do max
                answer = max(candidate , key=lambda x:(x[1], x[0]))[0]
                select.add(answer)
                result[templateKeyword] = answer

            return ''.join(map(lambda x:result.get(x, x), self.template[int(index)]['key']))
            
        index, templateKeywords = selectBestTemplate()
        return generate(index, templateKeywords, raw=True)

    def buildTopn(self):
        def doc2vec(wordCount):
            vec = np.zeros(400)
            for word, count in wordCount.items():
                vec += np.array(requests.get(self.apiDomain + '/kem/vector?keyword={}'.format(word)).json()['value']) * count
            return vec / sum(wordCount.values())


        def tfidf(context):
            tfidf = requests.post(self.apiDomain + '/tfidf/tfidf?flag=n', data={'doc':context}).json()
            return [(key, {'key':{key:1}, 'count':value}) for key, value in tfidf]

        def kcem(wordCount):
            docVec = doc2vec(wordCount)
            kcemList = requests.get(self.apiDomain + '/kcem/kcemList?keywords={}'.format('+'.join(wordCount))).json()
            ###doc2vec version####
            # result = requests.post(self.apiDomain + '/kcem?keyword={}'.format('+'.join(wordCount)), data={'counter':json.dumps(wordCount)}).json()
            ######################
            def countHypernym(kcemList):
                result = defaultdict(dict)
                for kcem in kcemList:
                    if not kcem['value']:
                        continue
                    hypernym = kcem['value'][0][0]
                    originKcemKey = kcem['key']
                    termFrequency = wordCount[originKcemKey]

                    # 把hypernym原始的查詢key給紀錄起來，他在文本中出現幾次也是
                    result[hypernym].setdefault('key', {}).setdefault(originKcemKey, termFrequency)
                    result[hypernym]['count'] = result[hypernym].setdefault('count', 0) + termFrequency
                return result
            result = countHypernym(kcemList)
            return sorted(result.items(), key=lambda x:-x[1]['count'])

        def kcemCluster(wordCount):
            def clustering(kcemList):
                def simpleUnion(clusters, unionList):
                    finalCluster = clusters[unionList[0]]
                    for cluster in clusters:
                        if cluster['groupIdx'] in unionList:
                            finalCluster['key'].update(cluster['key'])
                            finalCluster['hypernymSet'].update(cluster['hypernymSet'])
                    clusters = [cluster for cluster in clusters if cluster['groupIdx'] not in unionList[0:]] + [finalCluster]
                    for groupIdx, cluster in enumerate(clusters):
                        cluster['groupIdx'] = groupIdx
                    return clusters
                    
                clusters = []
                for kcem in kcemList:
                    # kcem最頂端的keyword是沒有hypernym的，其他太爛的字也沒有
                    if not kcem['value']:
                        continue

                    originKcemKey = kcem['origin']
                    termFrequency = wordCount[originKcemKey]
                    kcemDict = dict({originKcemKey:termFrequency}, **{hypernym:termFrequency for hypernym, possibility in kcem['value']})

                    insert = False
                    unionList = []
                    for cluster in clusters:
                        groupIdx = cluster['groupIdx']

                        # 除了keyword自身的hypernyms以外，keyword自身也包含在內，只要有overlap就歸在同一群
                        intersection = set(cluster['hypernymSet']).intersection(set(kcemDict))
                        if intersection:
                            insert = True
                            unionList.append(groupIdx)

                            cluster['key'].update({originKcemKey:termFrequency})
                            for hypernym in kcemDict:
                                cluster['hypernymSet'][hypernym] = cluster['hypernymSet'].setdefault(hypernym, 0) + termFrequency

                    if not insert:
                        # 要注意，因為kcem有套用ngram搜尋，所以kcem的key是可能會重複的喔!!!
                        clusters.append({'key':{originKcemKey:termFrequency}, 'hypernymSet':kcemDict, 'groupIdx':len(clusters)})
                    else:
                        # 如果有insert過，就要判斷是否需要union
                        if len(unionList) >= 2:
                            clusters = simpleUnion(clusters, unionList)

                return clusters

            docVec = doc2vec(wordCount)
            kcemList = requests.get(self.apiDomain + '/kcem/kcemList?keywords={}'.format('+'.join(wordCount))).json()
            # sorting and format
            result = {}
            for cluster in clustering(kcemList):
                cluster['hypernym'] = sorted(cluster['hypernymSet'].items(), key=lambda x:-x[1])[0][0]
                result[cluster['hypernym']] = {'key':{k:wordCount[k] for k in cluster['key']}}
                result[cluster['hypernym']]['count'] = sum([wordCount[k] for k in cluster['key']])
            return sorted(result.items(), key=lambda x:-x[1]['count'])

        def hybrid(wordCount, context):
            tfidfDict = dict(requests.post(self.apiDomain + '/tfidf/tfidf?flag=n', data={'doc':context}).json())

            def harmonic_mean(term, tfNormalized):
                return 2 * tfNormalized * tfidfDict[term] / (tfNormalized + tfidfDict[term])

            result = kcemCluster(wordCount)
            for hypernym, dictionary in result:
                total = dictionary['count']
                for term, tf in dictionary['key'].items():
                    dictionary['key'][term] = tfidfDict.get(term, 0)
            return result

        if os.path.isfile(self.output):
            return
        data = []

        for (dir_path, dir_names, file_names) in pyprind.prog_bar(list(os.walk(self.accessibility_log))):
            for file in file_names:
                filePath = os.path.join(dir_path,file)
                context = ''.join([i['context'] for i in json.load(open(filePath))])
                wordCount = Counter(rmsw(context, 'n'))

                # 如果wordCount為空
                # 代表Context Text經過stopword過濾後沒剩下任何字
                if not wordCount:
                    continue

                if self.mode == 'kcem':
                    data.append((kcem(wordCount), filePath))
                elif self.mode == 'tfidf':
                    data.append((tfidf(context), filePath))
                elif self.mode == 'kcemCluster':
                    data.append((kcemCluster(wordCount), filePath))
                elif self.mode == 'hybrid':
                    data.append((hybrid(wordCount, context), filePath))

        json.dump(data, open(self.output, 'w'))

if __name__ == '__main__':
    import sys, pyprind, subprocess
    mode = sys.argv[1]

    b = Behavior2Text(mode)
    b.buildTopn()
    for index, (refineData, fileName) in enumerate(json.load(open(b.output, 'r'))):
        topn = b.getTopN(refineData, b.topNum) if refineData else []

        if not topn:
            print([])
        else:
            print(b.sentence(topn))