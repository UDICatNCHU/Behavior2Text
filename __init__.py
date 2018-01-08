import requests, os, json, threading
from collections import defaultdict, Counter
from scipy import spatial
import numpy as np

class Behavior2Text(object):
    def __init__(self):
        self.template = json.load(open('template.json', 'r'))
        self.accessibility_log = 'Human'
        # self.accessibility_log = 'HumanDev'
        self.output = 'result.json'
        self.outputContext = 'outputContext.json'
        self.EntityOnly = False
        # self.EntityOnly = True

    def sentence(self, topn):
        topn = dict(topn[:5])

        candidate = defaultdict(dict)
        for index, template in enumerate(self.template):
            for value in template['value']:
                candidate[str(index)].setdefault(template['key'][value], {})
                for topnKeyword in topn:
                    result = requests.get('http://udiclab.cs.nchu.edu.tw/kem/similarity?k1={}&k2={}'.format(template['key'][value], topnKeyword)).json()
                    if result == {}:
                        continue

                    candidate[str(index)][template['key'][value]][topnKeyword] = result['similarity']
                        
                candidate[str(index)]['sum'] = candidate[str(index)].setdefault('sum', 0) + max(candidate[str(index)][template['key'][value]].values(), default=0)

        # select most possible template to generate sentence
        index, top1 = sorted(candidate.items(), key=lambda x:-x[1]['sum'])[0]
        del top1['sum']

        def generate(topn, raw=False):
            select = set()
            result = {}

            for i in sorted(top1.items(), key=lambda x:max(x[1].items(), key=lambda x:x[1])[1], reverse=True):

                # 最好的情況是填入template的詞不要重複
                # 但是真的沒辦法也只能重複填了...
                best = [j for j in i[1].items() if j[0] not in select]
                if best == []:
                    best = [j for j in i[1].items()]

                answer = max(best , key=lambda x:x[1])[0]
                select.add(answer)
                result[i[0]] = answer

            # use raw data to generate sentence
            if raw:
                result = {k: sorted(topn[v]['key'].items(), key=lambda x:-x[1])[0][0] for k,v in result.items()}
            return ''.join(map(lambda x:result.get(x, x), self.template[int(index)]['key']))
        return generate(topn), generate(topn, raw=True)


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
                    docVec = doc2vec(wordCount)

                    if self.EntityOnly:
                        result = requests.post('http://udiclab.cs.nchu.edu.tw/kcem/countertopn?EntityOnly=true', data={'doc':json.dumps(wordCount)}).json()
                    else:
                        result = requests.post('http://udiclab.cs.nchu.edu.tw/kcem/countertopn', data={'doc':json.dumps(wordCount)}).json()
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
    for topn in json.load(open(b.output, 'r')):
        print(b.sentence(topn))