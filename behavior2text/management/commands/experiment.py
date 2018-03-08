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
        parser.add_argument('--percentageMax', type=int, default=100)
        parser.add_argument('--topNMax', type=int, default=3)
        parser.add_argument('--clusterTopnMax', type=int, default=3)
        parser.add_argument('--pic', type=str)

    @staticmethod
    def draw(NDCG_DICT, labels, pic):
        fig = plt.figure()
        fig.suptitle(pic, fontsize=14, fontweight='bold')
        ax = fig.add_subplot(111)
        ax.set_xlabel(pic)
        ax.set_ylabel('NDCG')
        colorList = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for key, value in sorted(NDCG_DICT.items()):
            plt.plot(labels[::5], value, 'o-', color=colorList.pop(),label=key)
        plt.legend(loc='best')
        plt.savefig('{}.png'.format(pic))

    def handle(self, *args, **options):

        NDCG_DICT = defaultdict(list)
        labels = []

        topNMax = options['topNMax']
        clusterTopnMax = options['clusterTopnMax']
        percentageMax = options['percentageMax']
        pic = options['pic']

        modeList = ['tfidf', 'CFN-Vertex-Weight', 'CFN-Vertex-Weight-TFIDF', 'CFN-Vertex-Degree', 'CFN-PageRank']

        def main(parameter, percentage, topN, clusterTopn):
            for method in modeList:
                b = Behavior2Text(method, topN, clusterTopn, percentage)
                b.buildTopn()
                ndcg = b.main()

                labels.append(parameter)
                NDCG_DICT[method].append(ndcg)

        if topNMax!=3 and clusterTopnMax!=3 and percentageMax!=100:
            for percentage in pyprind.prog_bar(list(range(1, percentageMax, 10))):
                for topN in range(1, topNMax):
                    for clusterTopn in range(1, clusterTopnMax):
                        main((percentage+topN+clusterTopn), percentage, topN, clusterTopn)
        elif topNMax!=3:
            percentage, clusterTopn = percentageMax, clusterTopnMax
            for topN in range(1, topNMax):
                main(topN, percentage, topN, clusterTopn)
        elif clusterTopnMax!=3:
            percentage, topN = percentageMax, topNMax
            for clusterTopn in range(1, clusterTopnMax):
                main(clusterTopn, percentage, topN, clusterTopn)
        else:
            topN, clusterTopn = topNMax, clusterTopnMax
            for percentage in list(range(1, percentageMax, 10)) + [100]:
                main(percentage, percentage, topN, clusterTopn)

        self.draw(NDCG_DICT, labels, pic)
        self.stdout.write(self.style.SUCCESS('finish !!!'))