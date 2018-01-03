import requests, os, json, threading
from udicOpenData.stopwords import rmsw
from collections import defaultdict, Counter
from json import JSONDecodeError

class Behavior2Text(object):
    def __init__(self):
        self.template = json.load(open('template.json', 'r'))
        self.accessibility_log = 'JSON'
        self.output = 'result.json'
        self.outputContext = 'outputContext.json'


    def sentence(self, log):
        # topn = json.load(open(file, 'r'))[:3]
        topn = log[:3]

        candidate = defaultdict(dict)
        for index, template in enumerate(self.template):
            for value in template['value']:
                candidate[str(index)].setdefault(template['key'][value], {})
                for topnKeyword in topn:
                    try:
                        result = requests.get('http://udiclab.cs.nchu.edu.tw/kem/similarity?k1={}&k2={}'.format(template['key'][value], topnKeyword[0])).json()
                    except JSONDecodeError as e:
                        continue

                    # Set keyword similarity threshold here
                    # if result['k2Similarity'] == 1:
                    #     candidate[str(index)][template['key'][value]][topnKeyword[0]] = result['similarity']
                    candidate[str(index)][template['key'][value]][topnKeyword[0]] = result['similarity']
                        
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
                topn = dict(topn)
                result = {k:'、'.join(topn[v]['key']) for k,v in result.items()}
            return ''.join(map(lambda x:result.get(x, x), self.template[int(index)]['key']))
        return generate(topn), generate(topn, raw=True)


    def buildTopn(self):
        '''
        use accessibility log to extract topN keyword
        '''
        if os.path.isfile('result.json'):
            return
        final = []
        contextList = []
        for (dir_path, dir_names, file_names) in os.walk(self.accessibility_log):
            for file in file_names:
                context = ''.join([i['context'] for i in json.load(open(os.path.join(dir_path,file)))])
                wordCount = Counter(rmsw(context, 'n'))
                result = requests.post('http://udiclab.cs.nchu.edu.tw/kcem/countertopn', data={'doc':json.dumps(wordCount)})
                if result.json() == []:
                    continue

                result = [i for i in result.json() if '消歧義' not in i[0]]
                final.append(result)
                contextList.append(context)

        json.dump(final, open(self.output, 'w'))
        json.dump(contextList, open(self.outputContext, 'w'))

if __name__ == '__main__':
    import sys, pyprind
    b = Behavior2Text()

    if sys.argv[1] == 'b':
        b.buildTopn()
    for i in pyprind.prog_bar(json.load(open('result.json', 'r'))):
        print(b.sentence(i))
