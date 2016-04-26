from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.contrib.sessions.models import Session
from django.contrib.auth.models import User
from django.contrib.sessions.backends.db import SessionStore
#from django.contrib.staticfiles.templatetags.staticfiles import static
from django.shortcuts import redirect
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse
from django.conf import *
import urllib
import numpy as np
import os
import re
from numpy import dot
from numpy.linalg import norm
import scipy.spatial.distance
from numpy import linalg as LA
from scipy import linalg,dot
from pymongo import MongoClient
client = MongoClient()
db = client.crawled

# Create your views here.

def index(request):
	template = loader.get_template('search.html')
	context = {}
	return HttpResponse(template.render(context, request))

def find(request):
	if request.method == 'GET':
		query = request.GET.get('query')
		s = SessionStore()
		db.sessionHistory.update({'session_key': s.session_key},{'$push': {"query": [query]}}, upsert=True)
		from sklearn.metrics.pairwise import cosine_similarity
		from sklearn.feature_extraction.text import TfidfVectorizer

		dic = passQuery(query)
		docs = dic['docs']

		ids = dic['ids']
		print len(ids)
		print len(ids)
		all_url = []
		urls = db.crawledScienceCollection.find()
		for url in urls:
			all_url.append(url)
		selected_url = []
		for i in range(0, len(ids)):
			selected_url.append(all_url[ids[i]]['url'])
		print selected_url
		tfidf_vectorizer = TfidfVectorizer()
		tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
		print tfidf_matrix.shape
		cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
		template = loader.get_template('results.html')
		context = {'docs': docs, 'url': selected_url, 'zip': zip(docs, selected_url)}
		return HttpResponse(template.render(context, request))

def rocchio(request):
	from sklearn.feature_extraction.text import CountVectorizer
	from sklearn.feature_extraction.text import TfidfVectorizer
	from sklearn.feature_extraction.text import TfidfTransformer
	from sklearn.utils.extmath import randomized_svd
	from sklearn import feature_selection
	import pandas as pd
	document_index = []
	s = SessionStore()
	sessionData = db.sessionHistory.find_one({"session_key":s.session_key})
	urls_visited = sessionData['url_visited']
	urls = []
	for url in urls_visited:
		urls.append(url[0])
	bodyContentList = db.crawledCollection.find({'url':{"$in":urls}}, {'body':1})
	body = []
	terms = []
	for x in bodyContentList:
		body.append(re.sub('[!@#$%^&*()[]./<>?\|`~-=_+]0-9', '', x['body']))

	# Turning the body content into a bag of words
	top_features=[]
	
	vectorizer = TfidfVectorizer(stop_words = 'english')
	X = vectorizer.fit_transform(body)
	indices = np.argsort(vectorizer.idf_)[::-1]
	features = vectorizer.get_feature_names()
	top_n = 10
	top_features.append([features[i] for i in indices[:top_n]])

	print top_features
	
	vectorizer = CountVectorizer(min_df = 1, stop_words = 'english')
	dtm = vectorizer.fit_transform(body)

	index=pd.DataFrame(dtm.toarray(),index=body,columns=vectorizer.get_feature_names())
	indexterms=vectorizer.get_feature_names()
	
	transform=TfidfTransformer()
	tfidf=transform.fit_transform(dtm)
	
	U, Sigma, V = randomized_svd(tfidf, n_components=5,
                                      n_iter=5, transpose=True,
                                      random_state=None)
	

	#getting the highes count of words and adding it into the query
	return HttpResponse(top_features)



def result(request):
	return HttpResponse("Result")

def redirect(request):
	s = SessionStore()
	url = request.GET.get('url')
	
	db.sessionHistory.update({'session_key': s.session_key},{'$push': {"url_visited": [url]}}, upsert=True)
	
	return HttpResponseRedirect(url)


# User defined functions

##############################
#Query Pre processing#
##############################

def passQuery(query):
	mtext = query.strip()
	print("mtext")
	print(mtext)
	docs =[]
	all_terms =[]

	if mtext != "":
		query_terms = queryParsing(mtext)
		
		print("Printing query terms...")
		print(query_terms)

		#fetching synonyms from wordnet
		for q in query_terms:
			synonyms = getSynonyms(q)
			for s in synonyms:
				all_terms.append(s)

		dup_removed_qterms = list(set(all_terms))
		
		indexTerms = np.load('C:\\Users\\ashiq\\Documents\\lsiSearch\\search\\IndexTerms.npy')

		print("indexTerms:")
		print(indexTerms)

		#finding similar terms form Vt vectors
		termIndexes=[i for i, x in enumerate(indexTerms.tolist()) if any(thing in x for thing in dup_removed_qterms)]
		related_terms = findSimilarTerms(termIndexes)

		# vectorizing query terms
		queryVector = np.zeros(indexTerms.size)

		for t in related_terms:
			queryVector[t] = 1

		print("query vector")
		print(queryVector.T)
		np.save("queryVector", queryVector.T)
		if LA.norm(queryVector) != 0:
			docs = execute_search(queryVector.T,termIndexes)
		else:
			docs = []
		return docs
		#ResultsPage.setDocuments(ResultsPage, docs)
		#ResultsPage.query.set(mtext)
		#controller.show_frame(ResultsPage)



def queryParsing(query):
	from whoosh import qparser
	from whoosh.analysis import RegexTokenizer
	from whoosh.lang.porter import stem
	from whoosh.lang.morph_en import variations
	from whoosh.analysis import StopFilter
	print("inside queryParsing")
	tokenizer = RegexTokenizer()
	return_list = []   
	
	#Removing stop words
	stopper = StopFilter()
	tokens = stopper(tokenizer(query))

	for t in tokens:
		
		#converting to lower case
		t.text = t.text.lower()
		
		#stemming
		s=stem(t.text)
		return_list.append(s)
		
		#adding variations
		termVariations = variations(t.text)
		for u in termVariations:
			return_list.append(u)

	return return_list


###########################################
#Search Execution#
###########################################
def execute_search(qvector,term_indexes):

	lsaMatrix = np.load('C:\\Users\\ashiq\\Documents\\lsiSearch\\search\\LSI_TFIDF_Matrix.npy')

	#transQueryVector = transformQuery(qvector)

	cosine_scores = cos_cdist(lsaMatrix,qvector)
	print("cosine_scores: ")
	print(cosine_scores)
	threshold = 5 #needs to be tuned with cross-validation

	#pick a max_similary_score. Dont retreieve the doc if its similary score is above that max_similary_score

	ids = np.array(cosine_scores).argsort()[:threshold]

	documents = np.load('C:\\Users\\ashiq\\Documents\\lsiSearch\\search\\documents.npy')
	docs=[]
	for i in range(0, len(ids)):
		docs.append(documents[ids[i]])

	dic = {'docs': docs, 'ids': ids}
	return dic

def cos_cdist(matrix, vector):
	v = vector.reshape(1, -1)
	return scipy.spatial.distance.cdist(matrix, v, 'cosine').reshape(-1)


def transformQuery(qvector):

	U = np.load('C:\\Users\\ashiq\\Documents\\lsiSearch\\search\\U_vectors.npy')
	Vt = np.load('C:\\Users\\ashiq\\Documents\\lsiSearch\\search\\Vt_vectors.npy')
	S = np.load('C:\\Users\\ashiq\\Documents\\lsiSearch\\search\\SigmaValues.npy')

	Sdiag = linalg.diagsvd(S, len(S), len(S))
	Sinv = LA.inv(Sdiag)

	for index in range(len(U) - 2, len(U)):
		Sinv[index] = 0


	transQuery = dot(dot(Sinv,U.T),qvector)
	return transQuery

def findSimilarTerms(term_indexes):
	threshold = 3
	similar_terms = []
	Vt_vectors = np.load('C:\\Users\\ashiq\\Documents\\lsiSearch\\search\\Vt_vectors.npy')
	for index in term_indexes:
		term = Vt_vectors.T[index]
		similarity_score = cos_cdist(Vt_vectors.T, term)
		term_ids = np.array(similarity_score).argsort()[:threshold]

		for t in range (0,len(term_ids)):
			similar_terms.append(term_ids[t])

	dup_removed = list(set(similar_terms))
	return dup_removed

#########################################
#WORD NET#
#########################################

def getSynonyms(text):
	from nltk.corpus import wordnet
	synonyms = []
	for syn in wordnet.synsets(text):
		for l in syn.lemmas():
			synonyms.append(l.name())
	return set(synonyms)

def getSimilarWords(text):
	from nltk.corpus import wordnet
	similarWords=[]
	for ss in wordnet.synsets(text):
		for sim in ss.similar_tos():
			similarWords.append(sim.name())
	return set(similarWords)