from django.conf.urls import url
from . import views

urlpatterns = [
	url(r'^$', views.index, name='index'),
	url(r'^result', views.result, name='result'),
	url(r'^find', views.find, name='find'),
	url(r'^rocchio', views.rocchio, name='rocchio'),
	url(r'^redirect', views.redirect, name='redirect'),
]