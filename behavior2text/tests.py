from django.core.management import call_command
from django.test import TestCase
from behavior2text import Behavior2Text

# Create your tests here.
class Behavior2TextTestCase(TestCase):        
    def test_experiment(self):
        args = []
        opts = {'topNMax':2, 'clusterTopnMax':2, 'percentageMax':2, 'pic':'test'}
        call_command('experiment', *args, **opts)
