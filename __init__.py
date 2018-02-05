import requests, os, json, threading, copy, math
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
        self.label = json.load(open('label.json', 'r'))

        self.DEBUG = True
        if self.DEBUG:
            self.apiDomain = 'http://140.120.13.243'
        else:
            self.apiDomain = 'http://udiclab.cs.nchu.edu.tw/'

        self.NDCG = 0

    @staticmethod
    def getTopN(topList, n):
        n = n if n < len(topList) else -1
        minCount = topList[n][1]['count']

        # 取到count數字不小於topN的topK，舉例來說：如果取top3，而第3個element的count是10，第4、5的count也都是10
        # 第6個element的count是9，則就取到第5個element
        # 再來檢查top5有沒有dictionary裏面的value是0的（tfidf or kcem的分數是0），是就剔除掉
        return [(hypernym, dictionary) for hypernym, dictionary in takewhile(lambda x:x[1]['count'] >= minCount, topList) if max(dictionary['key'].values(), key=lambda x:x) != 0][:n]

    def sentence(self, topn, fileName):
        def selectBestTemplate():
            def calTemplateSim(TemplateCandidate, templateIndex, template):
                for replaceIndex in template['replaceIndices']:
                    replaceWord = template['key'][replaceIndex]
                    TemplateCandidate[templateIndex].setdefault(replaceWord, {})

                    for _, topnKeywordDict in topn:
                        # 這邊的7是因為label.json最高會給到7個relevence的資料，所以就統一取7個，然後配合不同context決定 這比ndcg要比到top幾的relevence
                        topnKeywords = (x[0] for x in sorted(topnKeywordDict['key'].items(), key=lambda x:-x[1])[:3])
                        for topnKeyword in topnKeywords:
                            similarity = requests.get(self.apiDomain + '/kem/similarity?k1={}&k2={}'.format(replaceWord, topnKeyword)).json()
                            if similarity == {}:
                                similarity['similarity'] = 0
                            TemplateCandidate[templateIndex][replaceWord][topnKeyword] = similarity['similarity']
                    TemplateCandidate[templateIndex]['sum'] = TemplateCandidate[templateIndex].setdefault('sum', 0) + max(TemplateCandidate[templateIndex][replaceWord].values(), default=0) / len(template['replaceIndices'])

            TemplateCandidate = defaultdict(dict)
            for templateIndex, template in enumerate(self.template):
                calTemplateSim(TemplateCandidate, templateIndex, template)

            # select most possible template to generate sentence
            index, templateKeywords = sorted(TemplateCandidate.items(), key=lambda x:-x[1]['sum'])[0]
            del templateKeywords['sum']

            # refine to ndcg format
            finalTemplateKeywords = copy.deepcopy(templateKeywords)
            replaceWordList = list(templateKeywords.keys())
            for candidate in templateKeywords[replaceWordList[0]]:
                reserveKey, maxSim = '', 0
                for replaceWord in replaceWordList:
                    if templateKeywords[replaceWord][candidate] > maxSim:
                        maxSim = templateKeywords[replaceWord][candidate]
                        reserveKey = replaceWord

                for delKey in [i for i in replaceWordList if i != reserveKey]:
                    del finalTemplateKeywords[delKey][candidate]

            return index, finalTemplateKeywords

        def generate(index, templateKeywords, raw=False):
            def NDCG(answerTable, template):
                # get NDCG label data
                for i in self.label:
                    if i['file'] == fileName:
                        NDCG_labelData = i
                        break

                NDCG = 0
                for labelDataIndex, replaceIndex in enumerate(template["replaceIndices"]):
                    replaceWord = template['key'][replaceIndex]

                    if len(answerTable[replaceWord]) and str(labelDataIndex) in NDCG_labelData:
                        DCG = sum([(2**NDCG_labelData[str(labelDataIndex)].get(candidate, 0) - 1) / math.log(1+candidateIndex, 2) for candidateIndex, (candidate, value) in enumerate(answerTable[replaceWord], start=1)])
                        DCG_best = sum([(2**sorted(NDCG_labelData[str(labelDataIndex)].items(), key=lambda x:-x[1])[candidateIndex-1][1] - 1) / math.log(1+candidateIndex, 2) for candidateIndex in range(1, min(len(answerTable[replaceWord]), len(NDCG_labelData[str(labelDataIndex)]))+1)])
                        NDCG += DCG / DCG_best

                # len(template["replaceIndices"]) means how many blank we need to fill in this template
                # in each iteration, we'll calculate out a DCG / DCG_best, this is only a NDCG for a blank
                # and we need the NDCG for a sentence, so calculate the average need to divide with len(template["replaceIndices"]), aka blanks we have in this template.
                return NDCG / len(template["replaceIndices"]), NDCG_labelData

            answerTable = {}
            for templateKeyword, templateKeywordCandidateDict in templateKeywords.items():
                if not templateKeywordCandidateDict:
                    answerTable[templateKeyword] = [['X', 0]]
                else:
                    answerTable[templateKeyword] = sorted(templateKeywordCandidateDict.items(), key=lambda x:-x[1])

            ndcg_this_sentence, NDCG_labelData = NDCG(answerTable, self.template[int(index)])
            self.NDCG += ndcg_this_sentence
            return ''.join(map(lambda x:answerTable.get(x, x)[0][0] if type(answerTable.get(x, x)) == list and len(answerTable.get(x, x)) else x, self.template[int(index)]['key'])), ndcg_this_sentence, NDCG_labelData, self.template[int(index)]
            
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
            result = kcemCluster(wordCount)

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
    topnsFile = json.load(open(b.output, 'r'))
    for refineData, fileName in topnsFile:
        topn = b.getTopN(refineData, b.topNum) if refineData else []

        if not topn:
            print([])
        else:
            print(b.sentence(topn, fileName))
    print('NDCG is {}'.format(b.NDCG/len(topnsFile)))