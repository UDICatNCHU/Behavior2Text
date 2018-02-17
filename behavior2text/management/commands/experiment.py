from django.core.management.base import BaseCommand, CommandError
from behavior2text import Behavior2Text
from collections import defaultdict
import sys, pyprind, subprocess, json, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt #可視化模塊
import matplotlib.ticker as tick

class Command(BaseCommand):
    help = 'use this to test Behavior2Text !'

    def add_arguments(self, parser):
        # Positional arguments
        parser.add_argument('--topNMax', type=int, default=3)
        parser.add_argument('--clusterTopnMax', type=int, default=3)
        parser.add_argument('--accessibilityTopnMax', type=int, default=1)

    @staticmethod
    def draw(NDCG_DICT, labels):
        colorList = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for key, value in NDCG_DICT.items():
            plt.plot(range(0, len(value)), value, 'o-', color=colorList.pop(),label=key)
        plt.legend(loc='best')


        # You can specify a rotation for the tick labels in degrees or with keywords.
        plt.xticks(range(0, len(NDCG_DICT.values().__iter__().__next__())), labels, rotation='vertical')
        # plt.gca().yaxis.set_major_formatter(tick.FormatStrFormatter('%f ndcg'))
        plt.savefig('behavior2text.png')

    def handle(self, *args, **options):
        NDCG_DICT = defaultdict(list)
        labels = []

        topNMax = options['topNMax']
        clusterTopnMax = options['clusterTopnMax']
        accessibilityTopnMax = options['accessibilityTopnMax']

        modeList = ['tfidf', 'kcem', 'kcemCluster', 'hybrid', 'contextNetwork', 'pagerank']
        for accessibilityTopn in pyprind.prog_bar(list(range(0, accessibilityTopnMax, 2))):
            for topNum in range(1, topNMax, 2):
                for clusterTopn in range(1, clusterTopnMax, 2):
                    labels.append('{}-{}-{}'.format(accessibilityTopn, topNum, clusterTopn))
                    for mode in modeList:

                        print("accessibilityTopn:{} topn: {}, clusterTopn:{} method:{}".format(accessibilityTopn, topNum, clusterTopn, mode))
                        print('======================================')
                        b = Behavior2Text(mode, topNum, clusterTopn)
                        b.buildTopn(accessibilityTopn)
                        topnsFile = json.load(open(b.output, 'r'))
                        for refineData, fileName in topnsFile:
                            topn = b.getTopN(refineData, b.topNum) if refineData else []

                            if not topn:
                                print([])
                            else:
                                print(b.sentence(topn, fileName))

                        result = b.NDCG/len(topnsFile)
                        NDCG_DICT[mode].append(result)
                        print('NDCG is {}'.format(result))

                    for mode in modeList:
                        os.remove(mode + '.json')

        self.draw(NDCG_DICT, labels)
        self.stdout.write(self.style.SUCCESS('finish !!!'))

if __name__ == '__main__':
    c = Command()
    c.handle()