from __future__ import unicode_literals

from django.db import models
from mongoengine import *
# Create your models here.
class crawledCollection(Document):
	url =  StringField(required=True)
	domain =  StringField(required=True)
	body = StringField(required=True)
	title = StringField(required=True)