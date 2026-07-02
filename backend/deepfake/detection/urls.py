from django.urls import path

from .views import show_video, upload_video


urlpatterns = [
    path("upload/", upload_video, name="upload_video"),
    path("show-video/", show_video, name="show_video"),
]
