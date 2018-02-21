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
        parser.add_argument('--accessibilityTopnMax', type=int, default=0)
        parser.add_argument('--topNMax', type=int, default=3)
        parser.add_argument('--clusterTopnMax', type=int, default=3)
        parser.add_argument('--pic', type=str)

    @staticmethod
    def draw(NDCG_DICT, labels, pic):
        colorList = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for key, value in NDCG_DICT.items():
            plt.plot(labels[::6], value, 'o-', color=colorList.pop(),label=key)
        plt.legend(loc='best')
        plt.savefig('{}.png'.format(pic))

    def handle(self, *args, **options):

        NDCG_DICT = defaultdict(list)
        labels = []

        topNMax = options['topNMax']
        clusterTopnMax = options['clusterTopnMax']
        accessibilityTopnMax = options['accessibilityTopnMax']
        pic = options['pic']

        modeList = ['tfidf', 'kcem', 'kcemCluster', 'hybrid', 'contextNetwork', 'pagerank']

        def main(parameter, accessibilityTopn, topN, clusterTopn):
            for method in modeList:
                b = Behavior2Text(method, topN, clusterTopn, accessibilityTopn)
                b.buildTopn()
                ndcg = b.main()

                labels.append(parameter)
                NDCG_DICT[method].append(ndcg)

        if topNMax!=3 and clusterTopnMax!=3 and accessibilityTopnMax!=0:
            for accessibilityTopn in pyprind.prog_bar(list(range(0, accessibilityTopnMax))):
                for topN in range(1, topNMax):
                    for clusterTopn in range(1, clusterTopnMax):
                        main((accessibilityTopn+topN+clusterTopn), accessibilityTopn, topN, clusterTopn)
        elif topNMax!=3:
            accessibilityTopn, clusterTopn = accessibilityTopnMax, clusterTopnMax
            for topN in range(1, topNMax):
                main(topN, accessibilityTopn, topN, clusterTopn)
        elif clusterTopnMax!=3:
            accessibilityTopn, topN = accessibilityTopnMax, topNMax
            for clusterTopn in range(1, clusterTopnMax):
                main(clusterTopn, accessibilityTopn, topN, clusterTopn)
        elif accessibilityTopnMax!=0:
            topN, clusterTopn = topNMax, clusterTopnMax
            for accessibilityTopn in range(1, accessibilityTopnMax):
                main(accessibilityTopn, accessibilityTopn, topN, clusterTopn)

        self.draw(NDCG_DICT, labels, pic)
        self.stdout.write(self.style.SUCCESS('finish !!!'))