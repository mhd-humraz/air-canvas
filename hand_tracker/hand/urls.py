
from django.urls import path, include

from hand.hand_tracker.hand import admin
from . import views

urlpatterns = [
     
    path('', views.index, name='index'),
]
