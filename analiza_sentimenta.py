
import pandas as pd
df = pd.read_csv("restaurantReviews.txt", delimiter='\t', quoting=3)
#print(df.head())
#print(df.shape)

import nltk
import re
#nltk.download('stopwords')
#from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

podaci = []
sve_reci = []

for i in range(0,1000):

  # Izbacivanje specijalnih karaktera
  
  komentar = re.sub(pattern='[^a-zA-Z]',repl=' ', string=df['Review'][i])

  komentar = komentar.lower()
  
  reci = komentar.split()
  
  #reci = [rec for rec in reci if not rec in set(stopwords.words('english'))]

  # Manipulacija recima (izbacivanje nastavaka)
  
  ps = PorterStemmer()
  komentar = [ps.stem(rec) for rec in reci]
  
  # Spajanje reci
  komentar = ' '.join(komentar)

  # Dodavanje komentara u niz

  podaci.append(komentar)

  #za proveru broja reci
  #for rec in reci:
   # sve_reci.append(rec)



#print(podaci[0:10])
#print(len(set(sve_reci)))


# transformacija podataka  
from sklearn.feature_extraction.text import CountVectorizer
#uzima 1500 reci iz skupa koje se najcesce pojavljuju
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(podaci).toarray()
y = df.iloc[:, 1].values
 
#print(cv.get_feature_names_out())
#print(X.shape)
#print(y.shape)


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
#print(X_train.shape) # (800, 1500)#
#print(y_train.shape) #(800,)
#print(X_test.shape) #(200, 1500)
#print(y_test.shape) #(200,)
#print(y_pred.shape) #(200, )


# Tacnost
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
tacnost = accuracy_score(y_test,y_pred)
preciznost = precision_score(y_test,y_pred)
odziv = recall_score(y_test,y_pred)


#print("Tačnost: {}%".format(round(tacnost ,2) * 100))
#print("Preciznost: {}%".format(round(preciznost,2)*100))


import numpy as np
#from sklearn.metrics import classification_report

#print(classification_report(y_test, y_pred))

# Izbor hiperparametra
"""
najveca_tacnost = 0.0
alfa = 0.0
for i in np.arange(0.1,1.1,0.1):
  temp_classifier = MultinomialNB(alpha=i)
  temp_classifier.fit(X_train, y_train)
  temp_y_pred = temp_classifier.predict(X_test)
  tacnost = accuracy_score(y_test, temp_y_pred)
  print("Tačnost za alfa={} je: {}%".format(round(i,1), round(tacnost*100,2)))
  if tacnost>najveca_tacnost:
    najveca_tacnost = tacnost
    alfa = i

print('Najveća tacnost {}% je postignuta uz vrednost alfa = {}'.format(round(najveca_tacnost * 100, 2), round(alfa,1)))
"""
classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)


def sentiment(test_komentar):
  
  test_komentar = re.sub(pattern='[^a-zA-Z]',repl=' ', string = test_komentar)
  test_komentar = test_komentar.lower()
  test_komentar_reci = test_komentar.split()
  #test_komentar_reci = [rec for rec in test_komentar_reci if not rec in set(stopwords.words('english'))]
  
  ps = PorterStemmer()
  test = [ps.stem(rec) for rec in test_komentar_reci]
  test = ' '.join(test)

  tmp = cv.transform([test]).toarray()
  return classifier.predict(tmp)


test_primeri = ['Very delicious food', 'The food tasted pretty bad',"I didn't like the enviroment",'The music was pretty']
for komentar in test_primeri:
  print(komentar + " - {}".format((int)(sentiment(komentar))))

"""test_primer = input("Unesite recenicu za testiranje(na engleskom)")
if(sentiment(test_primer)):
  print("Recenica je pozitivna.")
else:
  print("Recenica je negativna") 
"""