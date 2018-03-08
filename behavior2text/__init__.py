import requests, os, json, threading, copy, math, random
from collections import defaultdict, Counter
from scipy import spatial
from itertools import takewhile
from udicOpenData.stopwords import rmsw

from behavior2text.utils.hybrid import hybrid
from behavior2text.utils.kcem import kcem
from behavior2text.utils.tfidf import tfidf
from behavior2text.utils.kcemCluster import kcemCluster
from behavior2text.utils.contextNetwork import contextNetwork
from behavior2text.utils.pagerank import pagerankMain

class Behavior2Text(object):
    def __init__(self, mode, topN=3, topnKeywordNum=3, percentage=100):
        self.mode = mode
        self.topN = topN
        self.topnKeywordNum = topnKeywordNum
        self.percentage = percentage

        self.baseDir = os.path.dirname(os.path.abspath(__file__))
        self.accessibility_log = os.path.join(self.baseDir, 'inputData')
        self.template = json.load(open(os.path.join(self.baseDir, 'labelData', 'template.json'), 'r'))
        self.label = json.load(open(os.path.join(self.baseDir, 'labelData', 'label.json'), 'r'))
        self.output = '{}-{}.json'.format(self.mode, self.percentage)

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

    def sentence(self, refineData, fileName, DEBUG=False):
        def selectBestTemplate():
            def calTemplateSim(TemplateCandidate, templateIndex, template):
                for replaceIndex in template['replaceIndices']:
                    replaceWord = template['key'][replaceIndex]
                    TemplateCandidate[templateIndex].setdefault(replaceWord, {})

                    if self.mode == 'tfidf':
                        topn = self.getTopN(refineData, self.topN * self.topnKeywordNum) if refineData else []
                    else:
                        topn = self.getTopN(refineData, self.topN) if refineData else []
                    for _, topnKeywordDict in topn:
                        # 這邊的self.topnKeywordNum是這筆log要比到top幾的relevence candidate keywords
                        topnKeywords = (x[0] for x in sorted(topnKeywordDict['key'].items(), key=lambda x:-x[1])[:self.topnKeywordNum]) if type(list(topnKeywordDict['key'].values())[0]) != list else (x[0] for x in sorted(topnKeywordDict['key'].items(), key=lambda x:-len(x[1]))[:self.topnKeywordNum])
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
                NDCG_labelData = ''
                for i in self.label:
                    if i['file'] in fileName:
                        NDCG_labelData = i
                        break
                NDCG = 0
                for labelDataIndex, replaceIndex in enumerate(template["replaceIndices"]):
                    replaceWord = template['key'][replaceIndex]

                    if len(answerTable[replaceWord]) and str(labelDataIndex) in NDCG_labelData:

                        # 不管answerTable給了幾個candidate,答案我排幾個ndcg就要計 算到幾個，answerTable數量不夠的就是都拿零分
                        minLength = len(NDCG_labelData[str(labelDataIndex)])
                        answerTable[replaceWord] = answerTable[replaceWord][:minLength] if len(answerTable[replaceWord]) >= minLength else answerTable[replaceWord] + [['', 0] for _ in range(minLength-len(answerTable[replaceWord]))]
                        DCG = sum([(2**NDCG_labelData[str(labelDataIndex)].get(candidate, 0) - 1) / math.log(1+candidateIndex, 2) for candidateIndex, (candidate, value) in enumerate(answerTable[replaceWord], start=1)])
                        DCG_best = sum([(2**sorted(NDCG_labelData[str(labelDataIndex)].items(), key=lambda x:-x[1])[candidateIndex-1][1] - 1) / math.log(1+candidateIndex, 2) for candidateIndex in range(1, minLength+1)])
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

            if DEBUG:
                return ''.join(map(lambda x:answerTable.get(x, x)[0][0] if type(answerTable.get(x, x)) == list and len(answerTable.get(x, x)) else x, self.template[int(index)]['key'])), ndcg_this_sentence, NDCG_labelData, self.template[int(index)]
            else:
                return ''.join(map(lambda x:answerTable.get(x, x)[0][0] if type(answerTable.get(x, x)) == list and len(answerTable.get(x, x)) else x, self.template[int(index)]['key'])), ndcg_this_sentence
            
        index, templateKeywords = selectBestTemplate()
        return generate(index, templateKeywords, raw=True)

    def buildTopn(self):
        if os.path.isfile(self.output):
            return
        data = []

        for (dir_path, dir_names, file_names) in os.walk(self.accessibility_log):
            for file in file_names:
                filePath = os.path.join(dir_path, file)
                contextJson = json.load(open(filePath, 'r'))
                percentageIndex = len(contextJson) * self.percentage // 100
                context = ''.join([contextJson[index]['context'] for index in random.sample(range(len(contextJson)), percentageIndex)])
                wordCount = Counter((i[0] for i in rmsw(context, flag=True) if i[1] == 'n'))

                # 如果wordCount為空
                # 代表Context Text經過stopword過濾後沒剩下任何字
                if not wordCount:
                    continue

                if self.mode == 'kcem':
                    data.append((kcem(self.apiDomain, wordCount), filePath))
                elif self.mode == 'tfidf':
                    data.append((tfidf(self.apiDomain, context), filePath))
                elif self.mode == 'CFN-Vertex-Weight':
                    data.append((kcemCluster(self.apiDomain, wordCount), filePath))
                elif self.mode == 'CFN-Vertex-Weight-TFIDF':
                    data.append((hybrid(self.apiDomain, wordCount, context), filePath))
                elif self.mode == 'CFN-Vertex-Degree':
                    data.append((contextNetwork(self.apiDomain, wordCount), filePath))

        if self.mode == 'CFN-PageRank':
            pagerankMain(self.output)
            return
        json.dump(data, open(self.output, 'w'))

    def generateTopn(self, accessibilityLog):
        data = []

        context = ''.join([i['context'] for i in accessibilityLog])
        wordCount = Counter((i[0] for i in rmsw(context, flag=True) if i[1] == 'n'))

        # 如果wordCount為空
        # 代表Context Text經過stopword過濾後沒剩下任何字
        if not wordCount:
            json.dump([], open(self.output, 'w'))
            return

        if self.mode == 'kcem':
            data.append((kcem(self.apiDomain, wordCount), ''))
        elif self.mode == 'tfidf':
            data.append((tfidf(self.apiDomain, context), ''))
        elif self.mode == 'CFN-Vertex-Weight':
            data.append((kcemCluster(self.apiDomain, wordCount), ''))
        elif self.mode == 'CFN-Vertex-Weight-TFIDF':
            data.append((hybrid(self.apiDomain, wordCount, context), ''))
        elif self.mode == 'CFN-Vertex-Degree':
            data.append((contextNetwork(self.apiDomain, wordCount), ''))

        if self.mode == 'CFN-PageRank':
            pagerankMain()
            return
        json.dump(data, open(self.output, 'w'))

    def main(self, DEBUG=False):
        topnsFile = json.load(open(self.output, 'r'))
        for refineData, fileName in topnsFile:
            topn = self.getTopN(refineData, self.topN) if refineData else []
            if not topn:
                print([])
            else:   
                print(self.sentence(refineData, fileName, DEBUG))
        result = self.NDCG/len(topnsFile)
        print('{}: NDCG is {}'.format(self.mode, result))
        return result
        

if __name__ == '__main__':
    import sys, subprocess
    mode = sys.argv[1]
    DEBUG = True

    b = Behavior2Text(mode)
    b.buildTopn()
    b.main(DEBUG)