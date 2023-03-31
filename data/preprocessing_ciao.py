import csv
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
import pickle

with open("rating.txt", "r") as file:
    content = file.read()
    
data = content
attributes = data.split(":::</endperson>")
attributes = [a.strip().split("::::")[:7] for a in attributes]


data = attributes

with open('ciao_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)
    
data = pd.read_csv('ciao_data.csv',header=None, names=['user','product','category','rating','helpfulness','time','review_content'],low_memory=False)
df = data

df1= df[(df['rating'].isin(['10','20','40','50']))]
df1 = df1[df1['helpfulness'] != 'not helpful']
df1 = df1[df1['review_content'].apply(lambda x: len(x.split()) >= 10)]
df1['user'] = df1['user'].str.split(':').str[0]
df1['review_content'] = df1['review_content'].apply(lambda x: ' '.join(x.split()[:256]) + ' ... ' + ' '.join(x.split()[-256:]))
df1['rating'] = (df1['rating'].astype(float) / 10).astype(int)

convert_dict = {'user': int,
                'product': str,
                'category':str,
                'time': object,
                'rating': int,
                'helpfulness':str,
                'review_content':str
                }
 
df1 = df1.astype(convert_dict)

def rating_to_label(r):
    if r <= 2:
        return 0
    else:
        return 1
    
df1['review_content'] = df1.review_content.apply(lambda x: x.lower())
df1['label'] = df1['rating'].apply(rating_to_label)
df1['time'] = pd.to_datetime(df1['time'], format='%d.%m.%Y')
df1['year'] = df1['time'].dt.year
df1['month'] = df1['time'].dt.month
df1['day'] = df1['time'].dt.day
df1.drop_duplicates(subset=['review_content'], inplace=True)
df1.dropna(inplace=True)
df1.rename(columns={'review_content':'text'}, inplace=True)
df1.reset_index(inplace=True, drop=True)

df1 = df1[['user', 'time', 'year', 'month', 'day', 'text', 'rating', 'label']]

train_dev, test = train_test_split(df1, test_size=0.2, random_state=123, stratify=df1[['rating']])
train, dev = train_test_split(train_dev, test_size=0.125, random_state=123, stratify=train_dev[['rating']])

train.to_csv('ciao_train.csv', index=False)
dev.to_csv('ciao_dev.csv', index=False)
test.to_csv('ciao_test.csv', index=False)

edge_set = set()
users = set(df1.user)

with open("trustnetwork.txt", "r") as file:
    content1 = file.read()

data1 = content1
attributes1 = data1.split("::::\n")
attributes1 = [a.strip().split("::::") for a in attributes1]
data1 = attributes1

data1 = pd.DataFrame(data1)
data1.rename(columns={0:'userid',1:'friends'}, inplace=True)
data1 = data1.dropna()

convert_dict = {'userid': int,
                'friends':int
                }
 
data1 = data1.astype(convert_dict)

data2 = data1[data1['userid'].isin(users)]
data2 = data2[data2['friends'].isin(users)]

for index, row in data2.iterrows():
    userid = row["userid"]
    friend = row["friends"]
    edge_set.add((userid, friend))

with open('ciao_edges.p', 'wb') as f:
    pickle.dump(edge_set, f)
with open('ciao_users.p', 'wb') as f:
    pickle.dump(users, f)
    
