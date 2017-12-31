import requests, os, json, threading
from collections import defaultdict

class Behavior2Text(object):
    def __init__(self):
        self.template = json.load(open('template.json', 'r'))


    def sentence(self):

        topn = json.load(open('tmp.json', 'r'))[:3]

        candidate = defaultdict(dict)
        for index, template in enumerate(self.template):
            for value in template['value']:
                candidate[str(index)].setdefault(template['key'][value], {})
                for topnKeyword in topn:
                    result = requests.get('http://udiclab.cs.nchu.edu.tw/kem/similarity?k1={}&k2={}'.format(template['key'][value], topnKeyword[0])).json()
                    if result['k2Similarity'] == 1:
                        candidate[str(index)][template['key'][value]][topnKeyword[0]] = result['similarity']
                candidate[str(index)]['sum'] = candidate[str(index)].setdefault('sum', 0) + max(candidate[str(index)][template['key'][value]].values())

        # select most possible template to generate sentence
        index, top1 = sorted(candidate.items(), key=lambda x:-x[1]['sum'])[0]
        del top1['sum']
        def generate(top, topn, raw=False):
            select = set()
            result = {}

            for i in sorted(top1.items(), key=lambda x:max(x[1].items(), key=lambda x:x[1])[1], reverse=True):
                answer = max([j for j in i[1].items() if j[0] not in select], key=lambda x:x[1])[0]
                select.add(answer)
                result[i[0]] = answer

            # use raw data to generate sentence
            if raw:
                topn = dict(topn)
                result = {k:topn[v]['key'] for k,v in result.items()}
            return ''.join(map(lambda x:result.get(x, x), self.template[int(index)]['key']))
        return generate(top1, topn), generate(top1, topn, raw=True)


    def buildTopn(self):
        '''
        use accessibility log to extract topN keyword
        '''
        final = []
        def threadJob(file):
            doc = ''.join([i['context'] for i in json.load(file)])
            # print(doc)
            result = requests.post('http://udiclab.cs.nchu.edu.tw/kcem/topn', data={'doc':doc})
            final.append(result.json())
        for (dir_path, dir_names, file_names) in os.walk('JSON'):
            workers = [threading.Thread(target=threadJob, kwargs={'file':open(os.path.join(dir_path,file))}) for file in file_names]

        for thread in workers:
            thread.start()

        # Wait for all threads to complete
        for thread in workers:
            thread.join()

        json.dump(final, open('result.json', 'w'))

if __name__ == '__main__':
    b = Behavior2Text()
    print(b.sentence())
