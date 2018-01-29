import requests, os, json, threading
from collections import defaultdict, Counter
from scipy import spatial
import numpy as np
from itertools import takewhile
from udicOpenData.stopwords import rmsw

class Behavior2Text(object):
    def __init__(self, mode):
        self.template = json.load(open('template.json', 'r'))
        self.topNum = 5
        self.accessibility_log = 'goodHuman'
        self.mode = mode
        self.output = '{}.json'.format(self.mode)
        self.EntityOnly = False
        # self.EntityOnly = True

    def getTopN(self, topList, n):
        if self.mode == 'kcem':
            n = n if n < len(topList) else -1
            minCount = topList[n][1]['count']
            return list(takewhile(lambda x:x[1]['count'] >= minCount, topList))
        elif self.mode == 'tfidf':
            n = n if n < len(topList) else -1
            minCount = topList[n][1]
            return list(takewhile(lambda x:x[1] >= minCount, topList))
        if self.mode == 'kcemCluster':
            n = n if n < len(topList) else -1
            minCount = topList[n][1]['count']
            return list(takewhile(lambda x:x[1]['count'] >= minCount, topList))
        else:
            raise Exception

    def sentence(self, topn):
        def selectBestTemplate():
            TemplateCandidate = defaultdict(dict)
            for templateIndex, template in enumerate(self.template):
                templateIndex = str(templateIndex)
                for replaceIndex in template['replaceIndices']:
                    replaceWord = template['key'][replaceIndex]
                    TemplateCandidate[templateIndex].setdefault(replaceWord, {})

                    for topnKeyword, _ in topn:
                        similarity = requests.get('http://udiclab.cs.nchu.edu.tw/kem/similarity?k1={}&k2={}'.format(replaceWord, topnKeyword)).json()
                        if similarity == {}:
                            continue
                        TemplateCandidate[templateIndex][replaceWord][topnKeyword] = similarity['similarity']
                    TemplateCandidate[templateIndex]['sum'] = TemplateCandidate[templateIndex].setdefault('sum', 0) + max(TemplateCandidate[templateIndex][replaceWord].values(), default=0) / len(template['replaceIndices'])

            # select most possible template to generate sentence
            index, templateKeywords = sorted(TemplateCandidate.items(), key=lambda x:-x[1]['sum'])[0]
            del templateKeywords['sum']
            return index, templateKeywords

        def generate(topn, index, templateKeywords, raw=False):
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


            # use raw data to generate sentence
            if (self.mode == 'kcem' or self.mode == 'kcemCluster') and raw:
                # use multiple key as the same reason above
                topn = dict(topn)
                result = {templateKey: max(topn[concept]['key'].items(), key=lambda x:(-x[1], x[0]))[0] for templateKey,concept in result.items()}
            return ''.join(map(lambda x:result.get(x, x), self.template[int(index)]['key']))
            
        index, templateKeywords = selectBestTemplate()
        return generate(topn, index, templateKeywords, raw=True)

    def buildTopn(self):
        def doc2vec(wordCount):
            vec = np.zeros(400)
            for word, count in wordCount.items():
                vec += np.array(requests.get('http://udiclab.cs.nchu.edu.tw/kem/vector?keyword={}'.format(word)).json()['value']) * count
            return vec / sum(wordCount.values())


        def tfidf(context):
            tfidf = requests.post('http://udiclab.cs.nchu.edu.tw/tfidf/tfidf?flag=n', data={'doc':context}).json()
            return tfidf

        def kcem(wordCount):
            docVec = doc2vec(wordCount)
            kcemList = requests.get('http://udiclab.cs.nchu.edu.tw/kcem/kcemList?keywords={}'.format('+'.join(wordCount))).json()
            ###doc2vec version####
            # result = requests.post('http://udiclab.cs.nchu.edu.tw/kcem?keyword={}'.format('+'.join(wordCount)), data={'counter':json.dumps(wordCount)}).json()
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
                def union():
                    pass
                    
                clusters = []
                for kcem in kcemList:
                    # kcem最頂端的keyword是沒有hypernym的，其他太爛的字也沒有
                    if not kcem['value']:
                        continue

                    originKcemKey = kcem['origin']
                    termFrequency = wordCount[originKcemKey]
                    kcemDict = dict({originKcemKey:termFrequency}, **{hypernym:termFrequency for hypernym, possibility in kcem['value']})

                    insert = False
                    for cluster in clusters:
                        # 除了keyword自身的hypernyms以外，keyword自身也包含在內，只要有overlap就歸在同一群
                        intersection = set(cluster['hypernymSet']).intersection(set(kcemDict))
                        if intersection:
                            insert = True

                            cluster['key'].update({originKcemKey:termFrequency})
                            for hypernym in kcemDict:
                                cluster['hypernymSet'][hypernym] = cluster['hypernymSet'].setdefault(hypernym, 0) + termFrequency

                    if not insert:
                        # 要注意，因為kcem有套用ngram搜尋，所以kcem的key是可能會重複的喔!!!
                        clusters.append({'key':{originKcemKey:termFrequency}, 'hypernymSet':kcemDict})
                return clusters

            docVec = doc2vec(wordCount)
            kcemList = requests.get('http://udiclab.cs.nchu.edu.tw/kcem/kcemList?keywords={}'.format('+'.join(wordCount))).json()
            # sorting and format
            result = {}
            for cluster in clustering(kcemList):
                cluster['hypernym'] = sorted(cluster['hypernymSet'].items(), key=lambda x:-x[1])[0][0]
                result[cluster['hypernym']] = {'key':{k:wordCount[k] for k in cluster['key']}}
                result[cluster['hypernym']]['count'] = sum([wordCount[k] for k in cluster['key']])
            return sorted(result.items(), key=lambda x:-x[1]['count'])

        def hybrid():
            pass

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