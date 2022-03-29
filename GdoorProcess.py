# %% Gdoor Data
# os.system('pip install -r requirements.txt')
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from functions import *

path_files = "/home/davidg/Codes/R_Codes/Projects/BoE/ARO/Gdoor"
# path_files = "/home/davidg/Codes/R_Codes/Projects/BoE/ARO/Gdoor/Results"

# Load and Clean Glassdoor Data
gd_raw = pd.read_csv(os.path.join(path_files,'allreviews.csv'),encoding="ISO-8859-1")
gd_raw['date_review']= pd.to_datetime(gd_raw['date_review'])
gd_raw = gd_raw.set_index(keys='date_review').sort_index()
gd_raw = gd_raw[~pd.isnull(gd_raw.index)]
print(gd_raw.info())

# %%
col = ['overall_rating','cons']
df = gd_raw[col]
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
df.groupby('overall_rating').cons.count().plot.bar(ylim=0)

from io import StringIO
# df = df[pd.notnull(df['Consumer complaint narrative'])]
# df.columns = ['Product', 'Consumer_complaint_narrative']
# df['category_id'] = df['Product'].factorize()[0]
overall_rating_df = df[['Product', 'category_id']].drop_duplicates().sort_values('category_id')
# overall_rating = dict(category_id_df.values)
# id_to_category = dict(category_id_df[['category_id', 'Product']].values)
df.head()

overall_rating

# gd_raw['location'].unique()
a1 = plt.figure(figsize=(20,10))
plt;plt.axes()
sns.lineplot(data=gd_raw.reset_index(),x='date_review',y='overall_rating',ci=None)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.cons).toarray()
labels = df.overall_rating
features.shape


from sklearn.feature_selection import chi2
import numpy as np
N = 2
for Product, category_id in sorted(overall_rating.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(Product))
  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

# a1 = plt.figure(figsize=(20,10));plt.axes()
# sns.lineplot(data=gd_time.query("dir=='BBSCU'")[['overall_rating']].rolling(100).mean(),x='date_review',y='overall_rating',ci=None)
# sns.lineplot(data=gd_time.query("dir=='Regulator'")[['overall_rating']].rolling(100).mean(),x='date_review',y='overall_rating',ci=None)


# a1 = plt.figure(figsize=(20,10));plt.axes()
# sns.lineplot(data=gd_time.query("dir=='BBSCU'"),x='date_review',y='overall_rating',ci=None)


['Regulator', 'MUKDT', 'IBD', 'BBSCU']
# Forecast note based on text

# Set data
df = gd_raw[['overall_rating','work_life_balance','culture_values','career_opp','comp_benefits','senior_mgmt']].dropna()
df = df.groupby(['overall_rating']).mean()

# Make polar plot
polar_plot(df)

#%% Basic Regression
(gd_raw['work_life_balance']==1).sum()
sns.scatterplot(data=gd_raw,y='overall_rating',x='work_life_balance')

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np
# exp_1 = lambda x: np.expand_dims(x,0)
df_reg = gd_raw.dropna()
X = (df_reg[['work_life_balance','culture_values','career_opp','comp_benefits','senior_mgmt']].values)
y = (df_reg['overall_rating'].values)

clf = LogisticRegression(random_state=0).fit(X, y)
clf.predict(X[:2, :])
clf.predict_proba(X[:2, :])
clf.score(X, y)

#%% Text Analysis

# Import NLP package
import stanza # slow, but best package to use 
stanza.download('en')
stanza_nlp = stanza.Pipeline('en')

# Get the Text
sentence = ' '.join([x for x in gd_raw['cons'].iloc[:50]])
anno     = stanza_nlp(sentence) # Run the pipeline to annotate text

word_list, word_len, word_upos = [], [], []
for sentence in anno.sentences:
    for word in sentence.words:
        word_list.append(word.lemma)
        word_len.append(len(word.text))
        word_upos.append(word.upos)

word_df = {'word'   : word_list, 
           'length' : word_len, 
           'pos'    : word_upos
          }
word_df = pd.DataFrame(word_df)
word_df.head()

noun_df = word_df[(word_df['pos'] == 'NOUN') &  (word_df['length'] > 1)]
noun_df.head()

# focus on the top 30 words
top_nouns = noun_df['word'].value_counts().head(30)
top_nouns

top_words_pros = get_top_words(gd_raw.iloc[:50]['pros'])

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize = (10,10))
ax = sns.barplot(x = top_words_pros.values, y = top_words_pros.index, color = 'purple')
ax.set(xlabel='frequency of word');

named_ents = pd.DataFrame()
named_ents['text'] = [e.text for e in anno.ents]
named_ents['ent_type'] = [e.type for e in anno.ents]
named_ents

named_ent_types = named_ents['ent_type'].value_counts()
named_ent_types

plt.figure(figsize = (10,6))
ax = sns.barplot(x = named_ent_ty                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   pes.values, y = named_ent_types.index, color = 'purple')
ax.set(xlabel='frequency of entity type');

named_ent_text = named_ents[named_ents['ent_type'] == 'ORG']['text'].value_counts().head(10)
named_ent_text

import spacy
nlp_spacy = spacy.load("en_core_web_sm")
anno_spacy = nlp_spacy(pros)

spacy_named_entities_org = []

for ent in anno_spacy.ents:
    if ent.label_ == "ORG":
        spacy_named_entities_org.append(ent.text)

pd.Series(spacy_named_entities_org).value_counts().head(10)

fpc_verb_list = []
for ent in anno_spacy.ents:
    if ent.text == 'UBS' and ent.root.dep_ in ['nsubj'] and ent.root.head.pos_ == 'VERB':
        fpc_verb_list.append(ent.root.head.lemma_)

fpc_verb_list_counts = pd.Series(fpc_verb_list).value_counts()
fpc_verb_list_counts

from nltk.corpus import verbnet


import stanza
import numpy as np
# stanza.download('en') # download English model

nlp = stanza.Pipeline('en', processors = 'tokenize,sentiment') # initialize English neural pipeline
df  = gd_raw.iloc[:100]

all_sentiment_scores = []
for i, text in enumerate(df['pros']):
    doc = nlp(text)
    sentiment_score = []
    for sentence in doc.sentences:
        sentiment_score.append(sentence.sentiment)
    all_sentiment_scores.append(np.nanmean(sentiment_score))
    print(f"annotating {i+1} of {len(df)} documents")

df['annotated_sentiment_scores'] = all_sentiment_scores


sns.boxplot(x = 'overall_rating', y = 'annotated_sentiment_scores', data = df)

import stanza
# stanza.download('en') # download English model
nlp = stanza.Pipeline('en', processors = 'tokenize,sentiment') # initialize English neural pipeline
doc = gd_raw['cons'].iloc[1]
doc = nlp(doc)

subject_np_head_lemmas = []
for sent in doc.sentences:
    # get the auxiliaries 
    auxs = [v for v in sent if (v.pos_ == 'AUX') and v.dep_ in ['ROOT', 'ccomp', 'conj']]
    # now get the subject noun phrase head lemmas 
    for aux in auxs:
        if 'acomp' in [child.dep_ for child in aux.children]:
            subject_np_head = [child.lemma_ for child in aux.children if child.dep_ == 'nsubj']
            subject_np_head_lemmas.extend(subject_np_head)
subject_np_head_lemmas


nlp = stanza.Pipeline('en') 
doc = nlp('The actors were not great. The script was absolutely fantastic.')
for sentence in doc.sentences:
        print(sentence.sentiment)




# 7. Topic models - news data set
from nltk.corpus import reuters

corp = [reuters.raw(text) for text in reuters.fileids()]
len(corp)

nlp = spacy.load('en_core_web_lg')

def get_lemmas(text):
    doc = nlp(text)
    lemma_list = []
    for t in doc:
        if t.pos_ == 'NUM':
            lemma_list.append('NUMERAL')
        elif (t.pos_ not in ['PUNCT', 'SPACE']) and (not t.is_stop):
            lemma_list.append(t.lemma_.lower())
        else:
            pass
    return lemma_list


lemma_corp = []

for i, text in enumerate(corp):
    lemma_corp.append(get_lemmas(text))
    if i % 100 == 0:
        print(f'lemmatizing {i+1} of {len(corp)} docs')   

lemma_corp[1]

from gensim import corpora
lemma_list = corpora.Dictionary(lemma_corp)
corp_data = [lemma_list.doc2bow(text) for text in lemma_corp]

from gensim.models.ldamodel import LdaModel
import re

my_num_topics = 90 
# this might take some time
ldamod = LdaModel(corp_data, 
                  num_topics = my_num_topics, 
                  id2word=lemma_list, 
                  passes=2,
                  random_state = 123) #https://radimrehurek.com/gensim/models/ldamodel.html

tops = ldamod.show_topics(num_topics=90, num_words=10)

topic_words = {}
for topic, word in tops:
    topic_words[topic] = re.sub('[^A-Za-z ]+', '', word).split()[:10]
    if len(topic_words[topic]) < 10:
        reps = 10 - len(topic_words[topic])
        topic_words[topic].extend(['-'] * reps)

pd.DataFrame(topic_words).T


# Document Similarity
nlp = spacy.load('en_core_web_lg') # note, the _sm model doesn't ship with word vectors, so you'll need the _md or _lg model for this

doc0 = nlp('Alice likes football.')
doc1 = nlp("Alex likes tennis")
doc2 = nlp("Inflation is high.")
doc3 = nlp("Economic uncertainty is low.")  

doc_list = [doc0, doc1, doc2, doc3]

sim_dict = {}

for i, one_doc in enumerate(doc_list):
    sim_dict[i] = [one_doc.similarity(another_doc) for another_doc in doc_list]
    
sim_df = pd.DataFrame(sim_dict)

sns.heatmap(sim_df, cmap = 'viridis')

# Classification
X = gd_raw['cons']
y = gd_raw['overall_rating']
# split the data into a train/test split (better: train/dev/test split)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)

from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer()
X_train = vect.fit_transform(X_train)
X_test = vect.transform(X_test)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train,y_train)

# assess model performance
from sklearn.metrics import classification_report
preds = clf.predict(X_test)
print(classification_report(y_test, preds))

# import model 
from sklearn.svm import SVC
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# assess model performance
preds = clf.predict(X_test)
print(classification_report(y_test, preds))

# import model 
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
# fit model to training data 
clf.fit(X_train.toarray(), y_train)  #requires a dense array 
# assess model performance
preds = clf.predict(X_test.toarray())
print(classification_report(y_test, preds))
# import model 
from sklearn.neighbors import KNeighborsClassifier

#instantiate the model
clf = KNeighborsClassifier()
# fit model to training data 
clf.fit(X_train, y_train)
# assess model performance
preds = clf.predict(X_test)
print(classification_report(y_test, preds))

# import model 
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
# fit model to training data 
clf.fit(X_train, y_train)

# assess model performance
preds = clf.predict(X_test)
print(classification_report(y_test, preds))



mods = [LogisticRegression(), RandomForestClassifier(), SVC(kernel = 'linear', probability = True)]
mod_names = ['logit', 'rf', 'svm']
res = []
from sklearn.metrics import brier_score_loss # a proper scoring method 
for mod in mods:
    mod.fit(X_train, y_train)
    probs = mod.predict_proba(X_test)[:,1] #need probabilities for brier score
    res.append(brier_score_loss(y_test, probs))

pd.DataFrame({'model' : mod_names, 'brier_score' : res})    


# import the procedure class
from sklearn.model_selection import GridSearchCV
# instantiate 
gridclf = GridSearchCV(KNeighborsClassifier(), 
                       param_grid = [{'n_neighbors' : [num for num in range(1,100,5)]}], 
                       scoring='neg_brier_score', #use a proper scoring method, e.g. brier or log loss  
                       cv=10, #number of folds 
                       verbose=2, #controls how much output you get 
                       return_train_score=True) #set to True to diagnose overfitting (but slower)

gridclf.fit(X_train, y_train)  #for GridSearchCV this may take some time. 

res = {'train_score' : gridclf.cv_results_['mean_train_score'],
       'cross_val_score' : gridclf.cv_results_['mean_test_score']
      }                        

res = pd.DataFrame(res, index = [num for num in range(1,100,5)])
res

res.plot()
plt.axvline(gridclf.best_params_['n_neighbors'], color = 'gray', linestyle = '--')

# assess model performance
preds = gridclf.predict(X_test)
print(classification_report(y_test, preds)) #compare with the knn above 




#instantiate and fit the model to the training data 
clf = LogisticRegression().fit(X_train, y_train) 

# get the coefficients 
feature_names = vect.get_feature_names()
num_feats = len(feature_names)
mod_coefs = clf.coef_.reshape(num_feats,)
res = pd.DataFrame({'feature_name' : feature_names, 'coef' : mod_coefs})

res = res.sort_values('coef', ascending = False)

top_features = res.head(10).append(res.tail(10))
top_features = res.head(10).append(res.tail(10))

plt.figure(figsize = (10, 6))
sns.barplot(y = 'feature_name', x = 'coef', data = top_features, color = 'blue')