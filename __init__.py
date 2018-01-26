import requests, os, json, threading
from collections import defaultdict, Counter
from scipy import spatial
import numpy as np
from itertools import takewhile
from udicOpenData.stopwords import rmsw

class Behavior2Text(object):
    def __init__(self):
        self.template = json.load(open('template.json', 'r'))
        self.topNum = 5
        self.accessibility_log = 'goodHuman'
        self.output = 'result.json'
        self.EntityOnly = False
        # self.EntityOnly = True

    @staticmethod
    def getTopN(topList, n, mode):
        if mode == 'kcem':
            n = n if n < len(topList) else -1
            minCount = topList[n][1]['count']
            return list(takewhile(lambda x:x[1]['count'] >= minCount, topList))
        elif mode == 'tfidf':
            n = n if n < len(topList) else -1
            minCount = topList[n][1]
            return list(takewhile(lambda x:x[1] >= minCount, topList))
        else:
            raise Exception

    def sentence(self, topn, mode):
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

        def generate(topn, mode, raw=False):
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
            if mode == 'kcem' and raw:
                # use multiple key as the same reason above
                topn = dict(topn)
                result = {templateKey: max(topn[concept]['key'].items(), key=lambda x:(-x[1], x[0]))[0] for templateKey,concept in result.items()}
            return ''.join(map(lambda x:result.get(x, x), self.template[int(index)]['key']))
        return generate(topn, mode, raw=True)

    def buildTopn(self):
        def doc2vec(wordCount):
            vec = np.zeros(400)
            for word, count in wordCount.items():
                vec += np.array(requests.get('http://udiclab.cs.nchu.edu.tw/kem/vector?keyword={}'.format(word)).json()['value']) * count
            return vec / sum(wordCount.values())

        def kcem(wordCount):
            docVec = doc2vec(wordCount)
            if self.EntityOnly:
                result = requests.post('http://udiclab.cs.nchu.edu.tw/kcem/counterKCEM?EntityOnly=true', data={'counter':json.dumps(wordCount)}).json()
            else:
                result = requests.post('http://udiclab.cs.nchu.edu.tw/kcem/counterKCEM', data={'counter':json.dumps(wordCount)}).json()
            def disambiguate():
                # If result would has a key 全部消歧義頁面
                # means result has ambiguous keywords
                # so need some post-processing
                nonlocal result
                result = dict(result)
                for keyword, count in result.setdefault('全部消歧義頁面', {}).setdefault('key', {}).items():
                    concept = requests.post('http://udiclab.cs.nchu.edu.tw/kcem?keyword={}'.format(keyword), data={'docvec':json.dumps(docVec.tolist())}).json()['value']
                    concept = concept[0][0] if concept else keyword
                    result.setdefault(concept, {}).setdefault('key', {})[keyword] = count
                    result[concept]['count'] = result[concept].setdefault('count', 0) + count
                del result['全部消歧義頁面']
            disambiguate()
            return sorted(result.items(), key=lambda x:-x[1]['count'])

        def tfidf(context):
            tfidf = requests.post('http://udiclab.cs.nchu.edu.tw/tfidf/tfidf?flag=n', data={'doc':context}).json()
            return tfidf

        if os.path.isfile(self.output):
            return
        data = []

        for (dir_path, dir_names, file_names) in pyprind.prog_bar(list(os.walk(self.accessibility_log))):
            for file in file_names:

                context = ''.join([i['context'] for i in json.load(open(os.path.join(dir_path,file)))])
                wordCount = Counter(rmsw(context, 'n'))

                # 如果wordCount為空
                # 代表Context Text經過stopword過濾後沒剩下任何字
                if not wordCount:
                    continue

                data.append((kcem(wordCount), tfidf(context), os.path.join(dir_path,file)))

        json.dump(data, open(self.output, 'w'))

if __name__ == '__main__':
    import sys, pyprind, subprocess
    b = Behavior2Text()

    if sys.argv[1] == 'b':
        b.buildTopn()
        for index, (kcemResult, tfidfResult, fileName) in enumerate(json.load(open(b.output, 'r'))):
            kcemTopn, tfidfTopn = b.getTopN(kcemResult, b.topNum, 'kcem') if kcemResult else [], b.getTopN(tfidfResult, b.topNum, 'tfidf') if tfidfResult else []

            # for KCEM
            if not kcemTopn:
                print([], end=' ')
            else:
                print(b.sentence(kcemTopn, 'kcem'), end=' ')

            # for tfidf
            if not tfidfTopn:
                print([])
            else:
                print(b.sentence(tfidfTopn, 'tfidf'))
    # elif sys.argv[1] == 't':
    #     b.tfidfSentence()
    # else:
    #     for i in range(2, 9):
    #         b.topNum = i
    #         subprocess.call(['rm', 'result.json'])

    #         print(i)
    #         print('=============================================================')
    #         print('kcem:')
    #         b.buildTopn()
    #         for index, topn in enumerate(json.load(open(b.output, 'r'))):
    #             print(index, b.sentence(topn, 'kcem'))
    #         print('tfidf:')
    #         b.tfidfSentence()
    #         print('=============================================================')