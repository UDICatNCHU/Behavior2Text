from django.core.management.base import BaseCommand, CommandError
from behavior2text import Behavior2Text
import json

class Command(BaseCommand):
    help = 'use this to output sequence with accessibility provided by Behavior2Text package !'

    def add_arguments(self, parser):
        # Positional arguments
        parser.add_argument('--method', type=str, default='kcemCluster')
        parser.add_argument('--debug', type=bool)
        parser.add_argument('--accessibilityTopnMax', type=int)

    def handle(self, *args, **options):
        method = options['method']
        DEBUG = options['debug']
        accessibilityTopnMax = options['accessibilityTopnMax']


        b = Behavior2Text(method)
        b.buildTopn(accessibilityTopnMax)
        topnsFile = json.load(open(b.output, 'r'))
        for refineData, fileName in topnsFile:
            topn = b.getTopN(refineData, b.topNum) if refineData else []

            if not topn:
                self.stdout.write(self.style.SUCCESS('None'))
            else:
                self.stdout.write(self.style.SUCCESS(b.sentence(topn, fileName, DEBUG)))
        self.stdout.write(self.style.SUCCESS('NDCG is {}'.format(b.NDCG/len(topnsFile))))