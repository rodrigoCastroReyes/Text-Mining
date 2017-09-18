import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from TextProcessor import *
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer

sns.set_style("whitegrid")

def get_words(id_user,ngram,n_top):
	client = MongoClient('localhost', 27017)
	db = client['twitter_db']
	collection = db[id_user]
	tweets_iterator = collection.find().sort('created_at_date', 1)

	tp = TextProcessor()

	data = []

	for rawTweet in tweets_iterator:
		text = rawTweet['text']
		text = tp.tokenization(text)
		data.append(text)

	#representar el conjunto de tweets 'data' en un modelo 'bag of words'
	count_vect = TfidfVectorizer(use_idf=False,ngram_range=(ngram,ngram))
	#si ngram = 2, se agruparan las palabras del diccionario en duplas, ..
	#use_idf = False para obtener la frecuencia normalizada para cada palabra

	x_train_counts = count_vect.fit_transform(data)#ajustar el modelo
	dictionary = np.array(count_vect.get_feature_names())#obtener el diccionario
	bag_of_words = x_train_counts.todense().view(np.ndarray)
	
	likelihood_words = np.mean(bag_of_words,axis=0)#obtiene el vector fila
	
	indexes = np.argsort(likelihood_words)#obtiene lista de indices ordenadas de menor a mayor
	indexes = indexes[::-1]#convierte la lista de indices de mayor a menor

	words = dictionary[indexes][0:n_top]#obtiene el top n_top de palabras
	likelihood_words = likelihood_words[indexes][0:n_top]#obtiene las probabilidades para cada palabra 

	#construye un DataFrame para almacenar los datos y visualizar grafica
	data = [ [ words[i], likelihood_words[i] ]  for i in range(n_top) ]
	df = pd.DataFrame(data=data,columns=['words','likelihood'])
	#df.to_csv('distribucion_frases_' + id_user  + '.csv')
	ax = sns.barplot(x="words", y="likelihood", data=df)
	plt.show()
	
#Scripts para encontrar los ngrams con mayor probabilidad de ocurrencia en la cuenta MashiRafael

id = 'KarlaMoralesR'
n_grams = 2
n_tops = 10
get_words(id,n_grams,n_tops)