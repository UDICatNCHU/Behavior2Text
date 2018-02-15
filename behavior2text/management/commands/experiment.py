from django.core.management.base import BaseCommand, CommandError
from behavior2text import Behavior2Text
import sys, pyprind, subprocess, json, os

class Command(BaseCommand):
    help = 'use this to test Behavior2Text !'

    def handle(self, *args, **options):
        modeList = ['tfidf', 'kcem', 'kcemCluster', 'hybrid', 'contextNetwork', 'pagerank']

        for topNum in pyprind.prog_bar(list(range(3, 10, 3))):
            for clusterTopn in range(3, 10, 3):
                print("topn: {}, clusterTopn:{}".format(topNum, clusterTopn))
                print('======================================')
                for mode in modeList:
                    b = Behavior2Text(mode, topNum, clusterTopn)
                    b.buildTopn()
                    topnsFile = json.load(open(b.output, 'r'))
                    for refineData, fileName in topnsFile:
                        topn = b.getTopN(refineData, b.topNum) if refineData else []

                        if not topn:
                            print([])
                        else:
                            print(b.sentence(topn, fileName))
                    print('NDCG is {}'.format(b.NDCG/len(topnsFile)))

                for mode in modeList:
                    os.remove(mode + '.json')

        self.stdout.write(self.style.SUCCESS('finish !!!'))

if __name__ == '__main__':
    c = Command()
    c.handle()