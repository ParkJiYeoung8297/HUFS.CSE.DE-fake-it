from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.shortcuts import render


@api_view(['GET'])
def helloAPI(request):
    return Response({"message": "hello world!"})
     
def index(request):
    return render(request, 'index.html')
# view.py