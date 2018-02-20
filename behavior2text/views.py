# -*- coding: utf-8 -*-
from django.http import JsonResponse
from behavior2text import Behavior2Text
import json

def behavior2text(request):
    '''
    keywords: type array
    '''
    if request.POST and 'data' in request.POST:
        b = Behavior2Text(request.POST['method'], int(request.POST['topN']), int(request.POST['clusterTopn']))
        b.generateTopn(json.loads(request.POST['data']))
        topnsFile = json.load(open(b.output, 'r'))
        
        for refineData, fileName in topnsFile:
            topnData = b.getTopN(refineData, b.topN) if refineData else []

            if not topnData:
                return JsonResponse({}, safe=False)
            else:
                return JsonResponse({'sentence':b.sentence(topnData, fileName)[0], 'method':request.POST['method'], 'topN':request.POST['topN'], 'clusterTopn':request.POST['clusterTopn']}, safe=False)
    return JsonResponse({}, safe=False)