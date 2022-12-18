from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('<str:room>/', views.room, name='room'),
    path('checkview', views.checkview, name='checkview'),
    path('send', views.send, name='send'),
    path('getReplies', views.getReplies, name='getReplies'),
    path('getMessages/<str:room>/', views.getMessages, name='getMessages'),
    path('getSingleMessage/<str:room>/', views.getSingleMessage, name='getSingleMessage'),
    path('systemReply', views.systemReply, name='systemReply'),
]