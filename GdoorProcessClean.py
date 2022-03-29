# Gdoor Data
# %% Packages
import os
os.system('pip install -r requirements.txt')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import stanza 
stanza.download('en')

stanza_nlp = stanza.Pipeline('en')

from functions import *

# %% Load Data
path_files = "/home/davidg/Codes/R_Codes/Projects/BoE/ARO/Gdoor"
path_files = "/home/davidg/Codes/R_Codes/Projects/BoE/ARO/Gdoor/Results"

df = pd.read_csv(os.path.join(path_files,'all_reviews.csv'),encoding="ISO-8859-1")
df['date_review']= pd.to_datetime(df['date_review'])
df = df.set_index(keys='date_review').sort_index()
df = df[~pd.isnull(df.index)]
print(df.info())

df = df.sort_values(by='firm',ascending=False)

# Polar Plot Across Dimension
df_polar = df[['overall_rating','work_life_balance','culture_values','career_opp','comp_benefits','senior_mgmt']].dropna().groupby(['overall_rating']).mean()

# Make polar plot
polar_plot(df_polar)

# %% Bag of words
import wordcloud
wc = wordcloud.WordCloud(background_color="white", 
                        max_words=350, 
                        width=1000, 
                        height=600, 
                        random_state=1).generate(' '.join([x for x in df['pros']]+[x for x in df['cons']]))
plt.figure(figsize=(15,15))
plt.imshow(wc)
plt.axis("off")

# %% Word Count
pros       = ' '.join([x for x in df.iloc[:50]['pros']])
annot_pros = stanza_nlp(pros) # Run the pipeline to annotate text

cons       = ' '.join([x for x in df.iloc[:50]['cons']])
annot_cons = stanza_nlp(cons) # Run the pipeline to annotate text

annotated_pros = to_table(annot_pros)
annotated_cons = to_table(annot_cons)

count_words_pros = annotated_pros['word'].value_counts().iloc[:30]
count_words_cons = annotated_cons['word'].value_counts().iloc[:30]

splits_pros = [x for x in count_words_pros.index if x in list(set(count_words_cons.index)^set(count_words_pros.index))]
splits_cons = [x for x in count_words_cons.index if x in list(set(count_words_cons.index)^set(count_words_pros.index))]

plt.figure(figsize = (15,10))
plt.subplot(121)
ax = sns.barplot(x = count_words_pros[splits_pros].values, y = count_words_pros[splits_pros].index, color = 'purple')
ax.set(xlabel='pros');
plt.subplot(122)
ax = sns.barplot(x = count_words_cons[splits_cons].values, y = count_words_cons[splits_cons].index, color = 'purple')
ax.set(xlabel='cons');

# # %% Sentiment
# nlp   = stanza.Pipeline('en', processors = 'tokenize,sentiment')
# annot = nlp(' '.join([x for x in df['pros']]+[x for x in df['cons']])[:2000])

# subject_np_head_lemmas = []
# for sent in annot.sentences:
#     # get the auxiliaries 
#     auxs = [v for v in sent if (v.pos_ == 'AUX') and v.dep_ in ['ROOT', 'ccomp', 'conj']]
#     # now get the subject noun phrase head lemmas 
#     for aux in auxs:
#         if 'acomp' in [child.dep_ for child in aux.children]:
#             subject_np_head = [child.lemma_ for child in aux.children if child.dep_ == 'nsubj']
#             subject_np_head_lemmas.extend(subject_np_head)
# subject_np_head_lemmas

# %% Topics


# %% Predict Pros and Cons

# Make Dataset
PC = df.reset_index()[['pros','cons']].melt()
PC['score'] = 0
PC[['score']] = PC[['score']].where(PC['variable']=='cons',1)
PC = PC.sample(frac = 1)

X = PC['value']
y = PC['score']

# Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)
# Vectorize
from sklearn.feature_extraction.text import TfidfVectorizer
vect    = TfidfVectorizer()
X_train = vect.fit_transform(X_train)
X_test  = vect.transform(X_test)

# Load models
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()

clf.fit(X_train,y_train)

# assess model performance
from sklearn.metrics import classification_report
preds = clf.predict(X_test)
print(classification_report(y_test, preds))

# Importance Plots
feature_names = vect.get_feature_names()
num_feats = len(feature_names)
mod_coefs = clf.coef_.reshape(num_feats,)
res = pd.DataFrame({'feature_name' : feature_names, 'coef' : mod_coefs})
res = res.sort_values('coef', ascending = False)

top_features = res.head(10).append(res.tail(10))
top_features = res.head(10).append(res.tail(10))

plt.figure(figsize = (10, 6))
sns.barplot(y = 'feature_name', x = 'coef', data = top_features, color = 'blue')

# %% Time Series
df.info()

df = df.reset_index()
df['date'] = df['date_review'].apply(lambda x: x.strftime('%Y-%m'))
df.set_index('date', inplace=True)
df.index = pd.to_datetime(df.index)
df_agg = df[['firm','overall_rating']].pivot_table(values='overall_rating', index=['date'], columns='firm',  aggfunc='mean')
df_agg.rolling(4).mean().corr()

df['firm'].unique()
df['location'].unique()

# %% Plots
b1  = df.query("firm in ['danske_bank', 'virgin_money','amtrust_financial','legal_and_general_group']")
agg = df.select_dtypes(include=['int','float']).reset_index().groupby('date').mean()

rol = 10
plt.figure(figsize=(10,15))
ax1 = plt.subplot(711)
sns.lineplot(data=b1.rolling(rol).mean(),x=b1.index,y='overall_rating',hue=b1['firm'],ci=None,legend=False)
agg[['overall_rating']].rolling(rol).mean().plot(ax=ax1,linewidth=2)
ax2 = plt.subplot(712, sharex = ax1)
sns.lineplot(data=b1.rolling(rol).mean(),x=b1.index,y='culture_values',hue=b1['firm'],ci=None,legend=False)
agg[['overall_rating']].rolling(rol).mean().plot(ax=ax2,linewidth=2)
ax2 = plt.subplot(713, sharex = ax1)
sns.lineplot(data=b1.rolling(rol).mean(),x=b1.index,y='work_life_balance',hue=b1['firm'],ci=None,legend=False)
agg[['work_life_balance']].rolling(rol).mean().plot(ax=ax2,linewidth=2)
ax2 = plt.subplot(714, sharex = ax1)
sns.lineplot(data=b1.rolling(rol).mean(),x=b1.index,y='diversity_inclusion',hue=b1['firm'],ci=None,legend=False)
agg[['diversity_inclusion']].rolling(rol).mean().plot(ax=ax2,linewidth=2)
ax2 = plt.subplot(715, sharex = ax1)
sns.lineplot(data=b1.rolling(rol).mean(),x=b1.index,y='career_opp',hue=b1['firm'],ci=None,legend=False)
agg[['career_opp']].rolling(rol).mean().plot(ax=ax2,linewidth=2)
ax2 = plt.subplot(716, sharex = ax1)
sns.lineplot(data=b1.rolling(rol).mean(),x=b1.index,y='comp_benefits',hue=b1['firm'],ci=None,legend=False)
agg[['comp_benefits']].rolling(rol).mean().plot(ax=ax2,linewidth=2)
ax2 = plt.subplot(717, sharex = ax1)
sns.lineplot(data=b1.rolling(rol).mean(),x=b1.index,y='senior_mgmt',hue=b1['firm'],ci=None,legend=False)
agg[['senior_mgmt']].rolling(rol).mean().plot(ax=ax2,linewidth=2)
    
# %% Twitter 
import configparser
from twitter import Twitter, OAuth
import datetime
from itertools import compress
c = configparser.ConfigParser()
c.read('/home/davidg/Codes/Python_Codes/DG_conf.cfg')
# connect to tweets
t = Twitter(auth=OAuth(
c['twitter']['access_token'],
c['twitter']['access_secret_token'],
c['twitter']['api_key'],
c['twitter']['api_secret_key']),
retry=True)

date_start = '01-02-2021'
date_end   = '01-05-2021'

# get tweets
l = t.statuses.home_timeline(screen_name='imf',count=3000,)
# clean date
date_tweets  = map(lambda x: datetime.datetime.strptime(x['created_at'].replace("+0000",""),'%c'),l)
# filter date
idx_date     = list(map(lambda x: datetime.datetime.strptime(date_end,"%d-%m-%Y") > x > datetime.datetime.strptime(date_start,"%d-%m-%Y"),(date_tweets)))
# get tweets
tweet_sorted = list(compress(l,idx_date))
# Sel   ect specific
gd_sorted = df.query("firm=='imf' and index >= datetime.datetime.strptime(@date_start,'%d-%m-%Y') and index <=@datetime.datetime.strptime(@date_end,'%d-%m-%Y')")


# [t["text"] for t in tweet_sorted]

#%% 
commerz_bank = df.query("firm=='commerzbank'")
ticks = pd.read_csv("/home/davidg/Downloads/data").set_index("Date").fillna(0)
ticks.index = pd.to_datetime(ticks.index)+datetime.timedelta(days=1)

# commerz_bank = commerz_bank.join(ticks)

plt.figure()
plt.plot(commerz_bank[['overall_rating']].rolling(5).mean())
plt.plot(ticks[['Wages Working Condition Controversies Count']].rolling(5).mean())

# commerz_bank[['Wages Working Condition Controversies Count','overall_rating']].plot()

#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# import googlefinance

# from googlefinance import getQuotes
# import json
# print(json.dumps(getQuotes('AAPL'), indent=2))


from pandas_datareader import data
tickers = ['AAPL', 'MSFT', '^GSPC','CBK.DE']

# We would like all available data from 01/01/2000 until 12/31/2016.
start_date = '2010-01-01'
end_date = '2016-12-31'

# User pandas_reader.data.DataReader to load the desired data. As simple as that.
panel_data = data.DataReader(tickers, 'yahoo', start_date, end_date)

# panel_data.to_frame().head(9)
# Getting just the adjusted closing prices. This will return a Pandas DataFrame
# The index in this DataFrame is the major index of the panel_data.
close = panel_data['Close']

# Getting all weekdays between 01/01/2000 and 12/31/2016
all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')

# How do we align the existing prices in adj_close with our new set of dates?
# All we need to do is reindex close using all_weekdays as the new index
close = close.reindex(all_weekdays)

# Reindexing will insert missing values (NaN) for the dates that were not present
# in the original set. To cope with this, we can fill the missing by replacing them
# with the latest available price for each instrument.
close = close.fillna(method='ffill')

# Get the MSFT timeseries. This now returns a Pandas Series object indexed by date.
msft = close[['CBK.DE']]

# Calculate the 20 and 100 days moving averages of the closing prices
short_rolling_msft = msft.rolling(window=20).mean()
long_rolling_msft = msft.rolling(window=100).mean()

# Plot everything by leveraging the very powerful matplotlib package
fig, ax = plt.subplots(figsize=(16,9))
ax = plt.subplot(211)
ax.plot(msft.index, msft, label='CBK')
ax.plot(short_rolling_msft.index, short_rolling_msft, label='20 days rolling')
ax.plot(long_rolling_msft.index, long_rolling_msft, label='100 days rolling')

ax.set_xlabel('Date')
ax.set_ylabel('Adjusted closing price ($)')
ax.legend()
ax = plt.subplot(212)
plt.plot(commerz_bank[['overall_rating']].rolling(10).mean())