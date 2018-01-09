import requests, os, json, threading
from collections import defaultdict, Counter
from scipy import spatial
import numpy as np
from itertools import takewhile

class Behavior2Text(object):
    def __init__(self):
        self.template = json.load(open('template.json', 'r'))
        self.topNum = 5
        self.accessibility_log = 'Human'
        # self.accessibility_log = 'HumanDev'
        self.output = 'result.json'
        self.outputContext = 'outputContext.json'
        self.EntityOnly = False
        # self.EntityOnly = True

    @staticmethod
    def getTopN(topList, n):
        n = n if n < len(topList) else -1
        minCount = topList[n][1]['count']
        return takewhile(lambda x:x[1]['count'] >= minCount, topList)

    def sentence(self, topn):
        topn = dict(self.getTopN(topn, self.topNum))

        TemplateCandidate = defaultdict(dict)
        for index, template in enumerate(self.template):
            for value in template['value']:
                TemplateCandidate[str(index)].setdefault(template['key'][value], {})
                for topnKeyword in topn:
                    result = requests.get('http://udiclab.cs.nchu.edu.tw/kem/similarity?k1={}&k2={}'.format(template['key'][value], topnKeyword)).json()
                    if result == {}:
                        continue

                    TemplateCandidate[str(index)][template['key'][value]][topnKeyword] = result['similarity']
                        
                TemplateCandidate[str(index)]['sum'] = TemplateCandidate[str(index)].setdefault('sum', 0) + max(TemplateCandidate[str(index)][template['key'][value]].values(), default=0) / len(template['value'])

        # select most possible template to generate sentence
        index, templateKeywords = sorted(TemplateCandidate.items(), key=lambda x:-x[1]['sum'])[0]
        del templateKeywords['sum']

        def generate(topn, raw=False):
            select = set()
            result = {}

            for templateKeyword, templateKeywordCandidates in sorted(templateKeywords.items(), key=lambda x:max(x[1].items(), key=lambda x:x[1])[1], reverse=True):
                # 最好的情況是填入template的詞不要重複
                # 但是真的沒辦法也只能重複填了...
                candidate = [templateKeywordCandidate for templateKeywordCandidate in templateKeywordCandidates.items() if templateKeywordCandidate[0] not in select]
                if candidate == []:
                    candidate = [templateKeywordCandidate for templateKeywordCandidate in templateKeywordCandidates.items()]

                # use multiple key for max
                # Because in some case, candidate has same similarity
                # so need to use key length to do max
                answer = max(candidate , key=lambda x:(x[1], x[0]))[0]
                select.add(answer)
                result[templateKeyword] = answer

            # use raw data to generate sentence
            if raw:
                # use multiple key as the same reason above
                result = {templateKey: sorted(topn[concept]['key'].items(), key=lambda x:(-x[1], x[0]))[0][0] for templateKey,concept in result.items()}
            return ''.join(map(lambda x:result.get(x, x), self.template[int(index)]['key']))
        return generate(topn, raw=True)

    def tfidfSentence(self):
        def tfidfTopn(topList, n):
            n = n if n < len(topList) else -1
            minCount = topList[n][1]
            return takewhile(lambda x:x[1] >= minCount, topList)

        tfidfList = []

        for (dir_path, dir_names, file_names) in os.walk(self.accessibility_log):
            if dir_path.endswith('/IRI'):
                for file in file_names:
                    context = ''.join([i['context'] for i in json.load(open(os.path.join(dir_path,file)))])
                    tfidf = requests.post('http://udiclab.cs.nchu.edu.tw/tfidf/tfidf?flag=n', data={'doc':context}).json()
                    if not tfidf:
                        continue
                    tfidf = list(tfidfTopn(tfidf, self.topNum))
                    tfidfList.append(tfidf)
                    
                    TemplateCandidate = defaultdict(dict)
                    for index, template in enumerate(self.template):
                        for value in template['value']:
                            TemplateCandidate[str(index)].setdefault(template['key'][value], {})
                            for topnKeyword, _ in tfidf:
                                result = requests.get('http://udiclab.cs.nchu.edu.tw/kem/similarity?k1={}&k2={}'.format(template['key'][value], topnKeyword)).json()
                                if result == {}:
                                    continue

                                TemplateCandidate[str(index)][template['key'][value]][topnKeyword] = result['similarity']
                                    
                            TemplateCandidate[str(index)]['sum'] = TemplateCandidate[str(index)].setdefault('sum', 0) + max(TemplateCandidate[str(index)][template['key'][value]].values(), default=0)

                    # select most possible template to generate sentence
                    index, templateKeywords = sorted(TemplateCandidate.items(), key=lambda x:-x[1]['sum'])[0]
                    del templateKeywords['sum']
                    def generate():
                        select = set()
                        result = {}

                        for templateKeyword, templateKeywordCandidates in sorted(templateKeywords.items(), key=lambda x:max(x[1].items(), key=lambda x:x[1])[1], reverse=True):
                            # 最好的情況是填入template的詞不要重複
                            # 但是真的沒辦法也只能重複填了...
                            candidate = [templateKeywordCandidate for templateKeywordCandidate in templateKeywordCandidates.items() if templateKeywordCandidate[0] not in select]
                            if candidate == []:
                                candidate = [templateKeywordCandidate for templateKeywordCandidate in templateKeywordCandidates.items()]

                            # use multiple key for max
                            # Because in some case, candidate has same similarity
                            # so need to use key length to do max
                            answer = max(candidate , key=lambda x:(x[1], x[0]))[0]
                            select.add(answer)
                            result[templateKeyword] = answer

                        # use raw data to generate sentence
                        return ''.join(map(lambda x:result.get(x, x), self.template[int(index)]['key']))
                    print(generate())
        json.dump(tfidfList, open('tfidf.json', 'w'))


    def buildTopn(self):
        def doc2vec(wordCount):
            vec = np.zeros(400)
            for word, count in wordCount.items():
                vec += np.array(requests.get('http://udiclab.cs.nchu.edu.tw/kem/vector?keyword={}'.format(word)).json()['value']) * count
            return vec / sum(wordCount.values())


        from udicOpenData.stopwords import rmsw
        '''
        use accessibility log to extract topN keyword
        '''
        if os.path.isfile('result.json'):
            return
        final = []
        fileNameList = []

        for (dir_path, dir_names, file_names) in pyprind.prog_bar(list(os.walk(self.accessibility_log))):
            if dir_path.endswith('/IRI'):
                for file in file_names:
                    context = ''.join([i['context'] for i in json.load(open(os.path.join(dir_path,file)))])
                    wordCount = Counter(rmsw(context, 'n'))

                    # 如果wordCount為空
                    # 代表Context Text經過stopword過濾後沒剩下任何字
                    if not wordCount:
                        continue
                    docVec = doc2vec(wordCount)


                    if self.EntityOnly:
                        result = requests.post('http://udiclab.cs.nchu.edu.tw/kcem/countertopn?EntityOnly=true', data={'doc':json.dumps(wordCount)}).json()
                    else:
                        result = requests.post('http://udiclab.cs.nchu.edu.tw/kcem/countertopn', data={'doc':json.dumps(wordCount)}).json()

                    # 因為countertopn 是要ngram的similarity為1才會做轉換
                    # 如果傳到api的counter沒有任何一個字的ngram similarity為1的話，就會回傳[]
                    if result == []:
                        continue

                    # Because result would has a key 全部消歧義頁面
                    # these are all ambiguous keyword
                    # so need some post-processing
                    result = dict(result)
                    for keyword, count in result.setdefault('全部消歧義頁面', {}).setdefault('key', {}).items():
                        concept = requests.post('http://udiclab.cs.nchu.edu.tw/kcem/kcemDisambiguous?keyword={}'.format(keyword), data={'docvec':json.dumps(docVec.tolist())}).json()['value']
                        concept = concept[0][0] if concept else keyword
                        result.setdefault(concept, {}).setdefault('key', {})[keyword] = count
                        result[concept]['count'] = result[concept].setdefault('count', 0) + count
                    del result['全部消歧義頁面']

                    final.append(sorted(result.items(), key=lambda x:-x[1]['count']))
                    fileNameList.append((os.path.join(dir_path,file), wordCount, context))

        json.dump(final, open(self.output, 'w'))
        json.dump(fileNameList, open(self.outputContext, 'w'))

if __name__ == '__main__':
    import sys, pyprind
    b = Behavior2Text()

    if sys.argv[1] == 'b':
        b.buildTopn()
        for index, topn in enumerate(json.load(open(b.output, 'r'))):
            print(index, b.sentence(topn))
    elif sys.argv[1] == 't':
        b.tfidfSentence()