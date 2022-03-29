# %%
# Presentation
# There is an upward trend in firms' ratings, is it also observed in the comments?
# Can we make domain specific grading tools using the different rubrics?

# %%
import os
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyrsistent import v
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib as mpl
from termcolor import colored
from sklearn.impute import SimpleImputer
mpl.rcParams.update(mpl.rcParamsDefault)

path_files = "/home/davidg/Codes/R_Codes/Projects/BoE/ARO/Gdoor"

# Load Glassdoor Data
df = pd.read_csv(os.path.join(path_files,'allreviews.csv'),encoding="ISO-8859-1")
df.index = pd.to_datetime(df['date_review'])
df = df[~pd.isnull(df.index)].drop(columns=['Unnamed: 0'])
df['all_text'] = df.cons+' '+df.pros
print(df.info())

# Get column names for each types 
types_str = df.select_dtypes(include='object').columns
types_int = df.select_dtypes(include=[int,float]).columns

# Fill missing according to types
df[types_str] = df[types_str].fillna('NA')
df[types_int] = SimpleImputer(strategy='mean').fit_transform(df[types_int])

# Main Columns
cols         = ['overall_rating','headline','all_text']
# Rating Columns
ratings_cols = ['overall_rating','work_life_balance','culture_values','diversity_inclusion','career_opp']


# %% Overview of ratings per topics
ratings_ts = df[['firm']+ratings_cols].groupby([pd.Grouper(freq='30D')]).mean().ffill()

f,ax = plt.subplots(1,1,figsize=(15,10))
ratings_ts.plot(ax=ax)

sns.catplot(data=ratings_ts)
plt.show()

#%% Show some examples for each grade
for i in range(1,6):
    tmp = df.query('overall_rating==@i').sample(1)
    print(colored(tmp.firm.values[0].upper(),'blue'),'\n'+tmp.headline.values[0],tmp.job_title.values[0],'\nHeadline: ',tmp.headline.values[0],colored('\nPros:','green'),tmp.pros.values[0],colored('\nCons:','red'),tmp.cons.values[0],'\nRating:',colored(tmp.overall_rating.values[0],'red'),'\n')

# %% Non-semantic features 
# Word Size Distribution
df.reset_index(drop=True, inplace=True)

# The length of pros and cons is an important factor to detect polarity in opinions
df['head_len'] = df['headline'].apply(lambda x:len(nltk.word_tokenize(x)))
df['pros_len'] = df['pros'].apply(lambda x:len(nltk.word_tokenize(x)))
df['cons_len'] = df['pros'].apply(lambda x:len(nltk.word_tokenize(x)))

bins_h = list(range(0,20,5))+[1000]
bins   = list(range(0,60,10))+[1000]
df['head_len'] = pd.cut(df['head_len'], bins = bins_h, labels=[str(i)+'-'+str(i+10) for i in bins_h[:-2]]+[str(bins_h[-2])+'+'])
df['pros_len'] = pd.cut(df['pros_len'], bins = bins, labels=[str(i)+'-'+str(i+10) for i in bins[:-2]]+[str(bins[-2])+'+'])
df['cons_len'] = pd.cut(df['cons_len'], bins = bins, labels=[str(i)+'-'+str(i+10) for i in bins[:-2]]+[str(bins[-2])+'+'])

f,ax = plt.subplots(3,1,figsize=(20,15))
ax[0] = sns.histplot(data=df,x='head_len',hue='overall_rating',multiple="dodge",ax=ax[0])
ax[0].set_xlabel('Headline Length (words)')

ax[1] = sns.histplot(data=df,x='pros_len',hue='overall_rating',multiple="dodge",ax=ax[1], legend=False)
ax[1].set_xlabel('Pros Length (words)')

ax[2] = sns.histplot(data=df,x='cons_len',hue='overall_rating',multiple="dodge",ax=ax[2], legend=False)
ax[2].set_xlabel('Cons Length (words)')

plt.show()

#%% Repartition by Overall Grade
fig = plt.figure(figsize=(8,5)) 
df.groupby('overall_rating')['headline'].count().plot.bar(ylim=0)
plt.show()

# %%
# Capture Topics for Headline (is it what we would expect ?)
from gensim.models import LdaModel
from nltk.tokenize import TreebankWordTokenizer
from gensim import corpora
import nltk
from string import punctuation
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

nltk.download('stopwords')
en_stop       = set(nltk.corpus.stopwords.words('english'))
to_be_removed = list(en_stop) + list(punctuation)

tok = TreebankWordTokenizer()
df['text'] = df.headline.apply(lambda x: x.lower())
# Tokenizing and removing stopwords
text_data  = list(df.text.apply(lambda x: list(filter(lambda a: a.lower() not in to_be_removed,tok.tokenize(x)))).array)
dictionary = corpora.Dictionary(text_data)
corpus     = [dictionary.doc2bow(text) for text in text_data]

ldamodel = LdaModel(corpus, id2word=dictionary, num_topics=4)

lda_display = gensimvis.prepare(ldamodel, corpus, dictionary)
pyLDAvis.display(lda_display)

#%%
# Study most common unigrams and bigrams for ratings and categoriess 
# Here we do Tfidf and use perfoms chi test to select the features with the highest values for the test chi-squared statistic from the tf-idf features.
from sklearn.feature_extraction.text import TfidfVectorizer

def get_ngrams(col, df, N=4):
    """This function captures unigrams and bigrams most specific to the text """
    tfidf    = TfidfVectorizer(sublinear_tf=True,min_df=5,norm='l2',encoding='latin-1',ngram_range=(1,2),stop_words='english')
    features = tfidf.fit_transform(df[col]).toarray()
    labels   = df.overall_rating

    # Terms correlated with the grades
    unigrams, bigrams = [], []
    for rating in set(labels):
        features_chi2 = chi2(features, labels == rating)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names_out())[indices]
        unigrams += [[v for v in feature_names if len(v.split(' '))==1][-N:]]
        bigrams  += [[v for v in feature_names if len(v.split(' ')) == 2][-N:]]
    return pd.DataFrame({'unigrams':unigrams,'bigrams':bigrams},index=set(labels))

cons_grams = get_ngrams('cons', df)
pros_grams = get_ngrams('pros', df)

grams = cons_grams.join(pros_grams,lsuffix='_cons',rsuffix='_cons')
# %%
# Print Most Typicals Words
for i in grams.itertuples():
    print(30*'-','\n','           Rating ',colored(i[0],'blue'),'\n',30*'-',colored('\nPros:','green'),'\n -Unigrams: ',', '.join(i[3]),'\n -Bigrams: ',', '.join(i[4]),colored('\nCons:','red'),'\n -Unigrams: ',', '.join(i[1]),'\n -Bigrams: ',', '.join(i[2]),sep='')

# %%
# Tf-idf Studies
# Here we use tfidf and logit regression to detect key unigrams and bigrams for the prediction of overall ratings
# Get tfidf
tfidf_cons    = TfidfVectorizer(min_df=5,ngram_range=(1,2),stop_words='english')
regs_cons     = tfidf_cons.fit_transform(df['cons'])
tfidf_pros    = TfidfVectorizer(min_df=5,ngram_range=(1,2),stop_words='english')
regs_pros     = tfidf_pros.fit_transform(df['pros'])
feature_names = np.array(tfidf_cons.get_feature_names())

def get_closest(query,cat,tfidf,regs,num=5):
    '''Get most similar words by tfidf.'''
    query_vec = tfidf.transform([query])
    results   = cosine_similarity(regs,query_vec)
    firsts_   = np.argsort(-results.flatten())[:num]
    return df[['overall_rating',cat]].iloc[firsts_].set_index('overall_rating',drop=True).sort_index()

# Some words can be treacherous as they are highly oriented but appear in opposite categories
get_closest('bad','cons',tfidf_cons,regs_cons)
get_closest('bad','pros',tfidf_pros,regs_pros)

get_closest('japanese','cons',tfidf_cons,regs_cons)
get_closest('japanese','pros',tfidf_pros,regs_pros)

get_closest('french','cons',tfidf_cons,regs_cons)
get_closest('french','pros',tfidf_pros,regs_pros)

# We search for the most relevant unigrams and bigrams
lr = LogisticRegression(penalty='l1',solver='liblinear',C=1,max_iter=200)
lr.fit(regs_cons,df.overall_rating)
 
# The coefficients
idx_coef = np.argsort(lr.coef_)

ratings     = idx_coef[0]
more_charac = feature_names[ratings][-10:]
less_charac = feature_names[ratings][:10]

print('For cons reviews:')
print('Most important words to detect rating:',', '.join(more_charac))
print('Least important words to detect rating:',', '.join(less_charac))

# %% 
# Add regression to detect key topics

# %%
# Here we construct a skipgram model using FastText
# from keras.preprocessing.text import Tokenizer
from gensim.models.fasttext import FastText
import numpy as np
import matplotlib.pyplot as plt
import nltk
from string import punctuation
from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
# from nltk import WordPunctTokenizer
import nltk
import re
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))
df_cons = '. '.join(df.cons+df.pros)

stemmer = WordNetLemmatizer()

def preprocess_text(document):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(document))
        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)
        # Converting to Lowercase
        document = document.lower()
        # Lemmatization
        tokens = document.split()
        tokens = [stemmer.lemmatize(word) for word in tokens]
        tokens = [word for word in tokens if word not in en_stop]
        tokens = [word for word in tokens if len(word) > 3]

        preprocessed_text = ' '.join(tokens)

        return preprocessed_text


embedding_size = 100
window_size = 40
min_word = 5
down_sampling = 1e-2

df_cons_text = sent_tokenize(df_cons)
final_corpus = [preprocess_text(sentence) for sentence in df_cons_text if sentence.strip() !='']
word_punctuation_tokenizer = nltk.WordPunctTokenizer()
word_tokenized_corpus = [word_punctuation_tokenizer.tokenize(sent) for sent in final_corpus]

model_cons = FastText(word_tokenized_corpus,
                      vector_size=embedding_size,
                      window=window_size,
                      min_count=min_word,
                      sample=down_sampling,
                      sg=1,
                      )

# %%
# Here we study word similiarity in 2Ds 
# This allows to spot 
from sklearn.decomposition import PCA
from itertools import chain
topics = ['balance','values','management','diversity','career']
cols   = ['r','b','g','y','k','o']
topn   = 10
similar_words = {words: [item[0] for item in model_cons.wv.most_similar([words], topn=topn)] for words in topics}
# Print most similar words
print('Here are the most similar for the different categories of rating:')
for k,v in similar_words.items():
    print('  - ',k+":"+str(v))

similar_words = sum([[k] + v for k, v in similar_words.items()], [])
word_vectors  = model_cons.wv[similar_words]

cols = list(chain(*[(topn+1)*[col] for col in cols[:len(topics)]]))

# Reduce dimension with PCA
pca        = PCA(n_components=2)
p_comps    = pca.fit_transform(word_vectors)
word_names = similar_words

# Plot words
plt.figure(figsize=(18, 10))
plt.scatter(p_comps[:, 0], p_comps[:, 1], c=cols)

for word_names, x, y in zip(word_names, p_comps[:, 0], p_comps[:, 1]):
    plt.annotate(word_names, xy=(x+0.06, y+0.03), xytext=(0, 0), textcoords='offset points')
plt.xticks([])
plt.yticks([])
plt.show()

# %%
# In what follow we propose Aspect Based Sentiment analysis (to be continued)
import aspect_based_sentiment_analysis as absa

nlp = absa.load()
text = ("We are great fans of Slack, but we wish the subscriptions "
        "were more accessible to small startups.")

slack, price = nlp(text, aspects=['slack', 'price'])
assert price.sentiment == absa.Sentiment.negative
assert slack.sentiment == absa.Sentiment.positive

import aspect_based_sentiment_analysis as absa

name = 'absa/classifier-rest-0.2'
model = absa.BertABSClassifier.from_pretrained(name)
tokenizer = absa.BertTokenizer.from_pretrained(name)
professor = absa.Professor(...)     # Explained in detail later on.
text_splitter = absa.sentencizer()  # The English CNN model from SpaCy.
nlp = absa.Pipeline(model, tokenizer, professor, text_splitter)

# Break down the pipeline `call` method.
task = nlp.preprocess(text=..., aspects=...)
tokenized_examples = nlp.tokenize(task.examples)
input_batch = nlp.encode(tokenized_examples)
output_batch = nlp.predict(input_batch)
predictions = nlp.review(tokenized_examples, output_batch)
completed_task = nlp.postprocess(task, predictions)
import aspect_based_sentiment_analysis as absa

recognizer = absa.aux_models.BasicPatternRecognizer()
nlp = absa.load(pattern_recognizer=recognizer)
completed_task = nlp(text=..., aspects=['slack', 'price'])
slack, price = completed_task.examples

absa.summary(slack)
absa.display(slack.review)

absa.summary(price)
absa.display(price.review)



import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer 
import stanfordnlp

def aspect_sentiment_analysis(txt, stop_words, nlp):
    
    txt = txt.lower() # LowerCasing the given Text
    sentList = nltk.sent_tokenize(txt) # Splitting the text into sentences

    fcluster = []
    totalfeatureList = []
    finalcluster = []
    dic = {}

    for line in sentList:
        newtaggedList = []
        txt_list = nltk.word_tokenize(line) # Splitting up into words
        taggedList = nltk.pos_tag(txt_list) # Doing Part-of-Speech Tagging to each word

        newwordList = []
        flag = 0
        for i in range(0,len(taggedList)-1):
            if(taggedList[i][1]=="NN" and taggedList[i+1][1]=="NN"): # If two consecutive words are Nouns then they are joined together
                newwordList.append(taggedList[i][0]+taggedList[i+1][0])
                flag=1
            else:
                if(flag==1):
                    flag=0
                    continue
                newwordList.append(taggedList[i][0])
                if(i==len(taggedList)-2):
                    newwordList.append(taggedList[i+1][0])

        finaltxt = ' '.join(word for word in newwordList) 
        new_txt_list = nltk.word_tokenize(finaltxt)
        wordsList = [w for w in new_txt_list if not w in stop_words]
        taggedList = nltk.pos_tag(wordsList)

        doc = nlp(finaltxt) # Object of Stanford NLP Pipeleine
    
        dep_node = []
        for dep_edge in doc.sentences[0].dependencies:
            dep_node.append([dep_edge[2].text, dep_edge[0].id, dep_edge[1]])

        for i in range(0, len(dep_node)):
            if (int(dep_node[i][1]) != 0):
                dep_node[i][1] = newwordList[(int(dep_node[i][1]) - 1)]

        featureList = []
        categories = []
        for i in taggedList:
            if(i[1]=='JJ' or i[1]=='NN' or i[1]=='JJR' or i[1]=='NNS' or i[1]=='RB'):
                featureList.append(list(i)) # For features for each sentence
                totalfeatureList.append(list(i)) # Stores the features of all the sentences in the text
                categories.append(i[0])

        for i in featureList:
            filist = []
            for j in dep_node:
                if((j[0]==i[0] or j[1]==i[0]) and (j[2] in ["nsubj", "acl:relcl", "obj", "dobj", "agent", "advmod", "amod", "neg", "prep_of", "acomp", "xcomp", "compound"])):
                    if(j[0]==i[0]):
                        filist.append(j[1])
                    else:
                        filist.append(j[0])
            fcluster.append([i[0], filist])
            
    for i in totalfeatureList:
        dic[i[0]] = i[1]
    
    for i in fcluster:
        if(dic[i[0]]=="NN"):
            finalcluster.append(i)
        
    return(finalcluster)

import stanza
# stanza.download('en') # download English model
nlp = stanza.Pipeline('en') # initialize English neural pipeline

stop_words = set(stopwords.words('english'))
txt = "The Sound Quality is great but the battery life is very bad."
txt = 'Inflation is very high altough there are only a few issues with this.'
print(aspect_sentiment_analysis(txt, stop_words, nlp))

ll = df.cons.tolist()

for l in ll[:10]:
    print(aspect_sentiment_analysis(l, stop_words, nlp))

# Output: [['soundquality', ['great']], ['batterylife', ['bad']]]
# %%
