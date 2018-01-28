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
            return index, templateKeywords

        index, templateKeywords = selectBestTemplate()
        del templateKeywords['sum']

        def generate(topn, raw=False):
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
            if self.mode == 'kcem' and raw:
                # use multiple key as the same reason above
                topn = dict(topn)
                result = {templateKey: max(topn[concept]['key'].items(), key=lambda x:(-x[1], x[0]))[0] for templateKey,concept in result.items()}
            return ''.join(map(lambda x:result.get(x, x), self.template[int(index)]['key']))
        return generate(topn, raw=True)

    def buildTopn(self):
        def doc2vec(wordCount):
            vec = np.zeros(400)
            for word, count in wordCount.items():
                vec += np.array(requests.get('http://udiclab.cs.nchu.edu.tw/kem/vector?keyword={}'.format(word)).json()['value']) * count
            return vec / sum(wordCount.values())

        def kcem(wordCount):
            docVec = doc2vec(wordCount)
            kcemList = requests.get('http://udiclab.cs.nchu.edu.tw/kcem/kcemList?keywords={}'.format('+'.join(wordCount))).json()
            ###doc2vec version####
            # result = requests.post('http://udiclab.cs.nchu.edu.tw/kcem?keyword={}'.format('+'.join(wordCount)), data={'counter':json.dumps(wordCount)}).json()
            ######################

            result = defaultdict(dict)
            for kcem in kcemList:
                if not kcem['value']:
                    continue
                hypernyn = kcem['value'][0][0]
                key = kcem['key']
                count = wordCount[key]

                # 把hypernym原始的查詢key給紀錄起來，他在文本中出現幾次也是
                print(result, key, hypernyn, kcem)
                result[hypernyn].setdefault('key', {}).setdefault(key, count)
                result[hypernyn]['count'] = result[hypernyn].setdefault('count', 0) + count
            return sorted(result.items(), key=lambda x:-x[1]['count'])

        def tfidf(context):
            tfidf = requests.post('http://udiclab.cs.nchu.edu.tw/tfidf/tfidf?flag=n', data={'doc':context}).json()
            return tfidf

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