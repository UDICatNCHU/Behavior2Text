from django.core.management import call_command
from django.test import TestCase
from behavior2text import Behavior2Text

# Create your tests here.
class Behavior2TextTestCase(TestCase):
    def test_sequence(self):
        args = []
        opts = {'method':'kcemCluster', 'debug':True}
        call_command('sentence', *args, **opts)
        
    def test_experiment(self):
        args = []
        opts = {'topNMax':2, 'clusterTopnMax':2, 'accessibilityTopnMax':0, 'pic':'test'}
        call_command('experiment', *args, **opts)
