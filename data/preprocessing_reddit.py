import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
import itertools
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import sqlite3
import os
import re
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')
import networkx as nx
from collections import defaultdict
from itertools import combinations
from scipy.spatial.distance import jaccard
import csv
import pickle
import math


sql_conn = sqlite3.connect('database.sqlite')
corpus = pd.read_sql("SELECT author, body, subreddit, created_utc FROM May2015 where LENGTH(body) > 30 AND LENGTH(body) < 250 LIMIT 15000000", sql_conn)

corpus = corpus.groupby('author').filter(lambda x: len(x['subreddit'].unique())==4)
corpus.rename(columns={'author': 'user'}, inplace=True)
corpus.rename(columns={'body': 'text'}, inplace=True)

corpus.reset_index()

corpus.to_csv('Reddit.csv',index=False)

def lowercase_column(file_path, column_name):
    df = pd.read_csv(file_path)
    df[column_name] = df[column_name].astype(str)
    df[column_name] = df[column_name].apply(lambda x: x.lower())
    df.to_csv(file_path, index=False)


def remove_special_characters(file_path, column_name):
    df = pd.read_csv(file_path)
    df[column_name] = df[column_name].astype(str)
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    df.to_csv(file_path, index=False)
    

def remove_chiandjap_characters(file_path, column_name):
    df = pd.read_csv(file_path)
    df[column_name] = df[column_name].astype(str)
    df[column_name] = df[column_name].apply(lambda x: re.sub(u'[\u4e00-\u9fff]+[\u3040-\u309f]+|[\u30a0-\u30ff]+|[\u4e00-\u9fff]+|[\uff00-\uff9f]+', '', x))
    df.to_csv(file_path, index=False)
    

def remove_stop_words(file_path, column_name):
    df = pd.read_csv(file_path)
    df[column_name] = df[column_name].astype(str)
    stopwords='a, I, of, to, in, it, is, be, as, at, so, we, he, by, or, on, do, if, me, my, up, an, go, no, us, am, the, and, for, are, but, not, you, all, any, can, had, her, was, one, our, out, day, get, has, him, his, how, man, new, now, old, see, two, way, who, boy, did, its, let, put, say, she, too, use'.split(', ')
    stop_words=set(stopwords)
    df[column_name] = df[column_name].apply(lambda x: " ".join([word for word in x.split() if word.lower() not in stop_words]))
    df.to_csv(file_path, index=False)

def remove_duplicates(file_path, column_names):
    df = pd.read_csv(file_path)
    df.dropna(subset=column_names, inplace=True)
    df.drop_duplicates(subset=column_names, inplace=True)
    df = df.dropna()
    df = df[df['user'] != '[deleted]']
    df = df[df['user'] != 'deleted']
    df.to_csv(file_path, index=False)




lowercase_column(column_name='text',file_path='Reddit.csv')
lowercase_column(column_name='user',file_path='Reddit.csv')
remove_special_characters(column_name='text',file_path='Reddit.csv')
remove_special_characters(column_name='user',file_path='Reddit.csv')
remove_chiandjap_characters(column_name='text',file_path='Reddit.csv')
remove_stop_words(column_name='user',file_path='Reddit.csv')
lowercase_column(column_name='subreddit',file_path='Reddit.csv')
remove_duplicates(column_names=['user','subreddit','text'],file_path='Reddit.csv')



df = pd.read_csv('Reddit.csv', dtype=str)


df["created_utc"] = pd.to_datetime(df["created_utc"], unit='s')
df['created_utc'] = df['created_utc'].astype(str).str.split().str[0]


# convert the created_utc column to datetime type
df['created_utc'] = pd.to_datetime(df['created_utc'])

mask = (df['created_utc'] >= '2015-05-01') & (df['created_utc'] <= '2015-05-31')
filtered_df = df[mask]

num_rows_to_select = 1147461  
random_indices = np.random.choice(filtered_df.index, size=num_rows_to_select, replace=False)


start_date = pd.to_datetime('2019-09-01')
end_date = pd.to_datetime('2020-04-30')
random_dates = pd.to_datetime(np.random.randint(start_date.value, end_date.value, size=num_rows_to_select), unit='ns')


df.loc[random_indices, 'created_utc'] = random_dates

df.to_csv('Reddit_Data.csv', index=False)

df1 = pd.read_csv('Reddit_Data.csv')

df1["created_utc"] = pd.to_datetime(df1["created_utc"])


# Extract day, month, and year from the datetime column
df1["day"] = df1["created_utc"].dt.day
df1["month"] = df1["created_utc"].dt.month
df1["year"] = df1["created_utc"].dt.year


df1 = df1.rename(columns={'created_utc':'time'})

df1['time'] = df1['time'].astype(str).str.split().str[0]

df1 = df1.rename(columns={'user':'author'})
df1 = df1.rename(columns={'subreddit':'user'})

df1['time'] = pd.to_datetime(df1['time'])


current_unique = len(df1['user'].unique())

desired_unique = 9500
rows_to_drop = round(len(df1) - (desired_unique / current_unique) * len(df1))

df1 = df1.drop(df1.sample(n=rows_to_drop).index)

df1.to_csv('Reddit_dataset.csv', index=False)


train_dev, test = train_test_split(df1, test_size=0.2, random_state=123)
train, dev = train_test_split(train_dev, test_size=0.125, random_state=123)

train.to_csv('reddit_train.csv', index=False)
dev.to_csv('reddit_dev.csv', index=False)
test.to_csv('reddit_test.csv', index=False)

edge_set = set()
users = set(df1.user)

subreddits = df1['user'].str.split(' ', expand=True).stack().str.strip().unique()
authors = df1['author'].str.split(',', expand=True).stack().str.strip().unique()


author_subreddits = df1.groupby('author')['user'].apply(lambda x: set(x.str.split(',').sum())).to_dict()



# input data: dictionary with authors as keys and subreddits as values
data = author_subreddits

# create a dictionary that maps subreddits to sets of authors
subreddit_authors = defaultdict(set)
for author, subreddits in data.items():
    for subreddit in subreddits:
        subreddit_authors[subreddit].add(author)


similarity = {}
subreddit_pairs = itertools.combinations(sorted(subreddit_authors.keys()), 2)
for s1, s2 in subreddit_pairs:
    sim = len(subreddit_authors[s1] & subreddit_authors[s2]) / len(subreddit_authors[s1] | subreddit_authors[s2])
    similarity[frozenset([s1, s2])] = sim

threshold = 0.001
#create sets of nodes and edges based on similarity and threshold
nodes = set(subreddit_authors.keys())
edges = set()
for (s1, s2), sim in similarity.items():
    if sim > threshold:
        edges.add((s1, s2))


with open('reddit_edges.p', 'wb') as f:
    pickle.dump(edges, f)
with open('reddit_users.p', 'wb') as f:
    pickle.dump(nodes, f)
    
    
