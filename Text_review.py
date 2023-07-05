# -*- coding: utf-8 -*-
import pandas as pd
# import numpy as np
from collections import Counter


df =pd.read_excel('/Users/vaibhavbajpai/Downloads/Case Study GG 11.xlsx',sheet_name=1)

df.info()

df['Review Text']= df['Review Text'].astype(str).str.lower()

from nltk.tokenize import RegexpTokenizer

regexp = RegexpTokenizer('\w+')

df['text_token']=df['Review Text'].apply(regexp.tokenize)

import nltk

nltk.download('stopwords')
nltk.download('punkt')


from nltk.corpus import stopwords

# Make a list of english stopwords
stopwords = nltk.corpus.stopwords.words("english")

# Extend the list with your own custom stopwords
my_stopwords = ['https']
stopwords.extend(my_stopwords)


df['text_token'] = df['text_token'].apply(lambda x: [item for item in x if item not in stopwords])
df.head(3)

df['text_string'] = df['text_token'].apply(lambda x: ' '.join([item for item in x if len(item)>2]))

df[['Review Text', 'text_token', 'text_string']].head()

all_words = ' '.join([word for word in df['text_string']])

tokenized_words = nltk.tokenize.word_tokenize(all_words)

from nltk.probability import FreqDist

fdist = FreqDist(tokenized_words)

print(fdist)

df['text_string_fdist'] = df['text_token'].apply(lambda x: ' '.join([item for item in x if fdist[item] >= 1 ]))

df[['Review Text', 'text_token', 'text_string', 'text_string_fdist']].head()

nltk.download('wordnet')
nltk.download('omw-1.4')


from nltk.stem import WordNetLemmatizer

wordnet_lem = WordNetLemmatizer()

df['text_string_lem'] = df['text_string_fdist'].apply(wordnet_lem.lemmatize)

df['is_equal']= (df['text_string_fdist']==df['text_string_lem'])

df.is_equal.value_counts()

all_words_lem = ' '.join([word for word in df['text_string_lem']])
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

words = nltk.word_tokenize(all_words_lem)
fd = FreqDist(words)

fd.most_common(3)

fd.tabulate(3)


# Obtain top 10 words
top_10 = fd.most_common(10)

# Create pandas series to make plotting easier
fdist = pd.Series(dict(top_10))


import seaborn as sns
sns.set_theme(style="ticks")

sns.barplot(y=fdist.index, x=fdist.values, color='blue');


nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
df['polarity'] = df['text_string_lem'].apply(lambda x: analyzer.polarity_scores(x))
df.tail(3)

# Change data structure
df = pd.concat(
    [df.drop(['Unnamed: 0', 'Clothing ID', 'Age', 'polarity'], axis=1), 
     df['polarity'].apply(pd.Series)], axis=1)
df.head(3)

# Create new variable with sentiment "neutral," "positive" and "negative"
df['sentiment'] = df['compound'].apply(lambda x: 'positive' if x >0 else 'neutral' if x==0 else 'negative')
df.head(4)


# Tweet with highest positive sentiment
df.loc[df['compound'].idxmax()].values

# Tweet with highest negative sentiment 
# ...seems to be a case of wrong classification because of the word "deficit"
df.loc[df['compound'].idxmin()].values


# Number of tweets 
sns.countplot(y='sentiment', 
             data=df, 
             palette=['#b2d8d8',"#008080", '#db3d13']
             );


# Boxplot
sns.boxplot(y='compound', 
            x='sentiment',
            palette=['#b2d8d8',"#008080", '#db3d13'], 
            data=df);