"""
URL configuration for deepfake project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from .views import helloAPI
from .views import index
from django.conf import settings
from django.conf.urls.static import static
from detection.views import upload_video
from detection.views import show_video


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index), # React 앱 메인 페이지
    path("hello/",helloAPI),
    path('upload/',upload_video),
    path('showVideo/',show_video),

]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
