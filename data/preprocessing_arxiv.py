import logging
import json
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

def version_year(x):
    if True:
        i=x[-1]
        t=int(i['created'].split(' ')[3])
        if t>=2001 or t<=2020:
            return [int(t),int(i['created'].split(' ')[1]),str(i['created'].split(' ')[2])]
    return -1,-1,-1

def authors(x):
    t=[]
    regex = re.compile('[^a-zA-Z]')
    for i in x:
        name=set()
        for j in i[:len(i)]:
            temp=regex.sub('',j.lower())
            if temp!='' and len(j)<15:
                name.add(temp)


        t.append(name)
    return t

def cat(x):
    t=[]
    regex = re.compile('[^(a-zA-Z|\-|\.)]')
    for i in x.split():
        t.append(regex.sub(' ',i.lower()).split())
    return t

freq_group_cat={}
def frequency_dict(x):
    global frequency_cat
    #print(x)
    #return
    for i in x:
        if i[0] in freq_group_cat:
            freq_group_cat[i[0]]+=1
        else:
            freq_group_cat[i[0]]=1

cat_auth_dict={}
def cat_auth(x):
    regex = re.compile('[^(a-zA-Z|\-|\.)]')
    #print(x[0])
    #return
    for i in x[0].split():
        t=(regex.sub('',i.lower()))
        if t in cat_auth_dict:
            cat_auth_dict[t].extend(x[1])
        else:
            cat_auth_dict[t]=x[1]

stop_3='a, I, of, to, in, it, is, be, as, at, so, we, he, by, or, on, do, if, me, my, up, an, go, no, us, am, the, and, for, are, but, not, you, all, any, can, had, her, was, one, our, out, day, get, has, him, his, how, man, new, now, old, see, two, way, who, boy, did, its, let, put, say, she, too, use'.split(', ')
stop_3=set(stop_3)
def abs_clean(x):
    regex = re.compile('[^(a-zA-Z|\.)]')
    t=''
    for i in x.split():
        s=regex.sub(' ',i.lower())
        if len(s)<=3 and s not in stop_3:
            continue
        t+=s
    return t

def abs_uncased(x):
    return x.lower()

def templ(x):
	return [i for l in x for i in l]


# docs = db.read_text('/kaggle/input/arxiv/arxiv-metadata-oai-snapshot.json').map(json.loads)
with open('arxiv-metadata-oai-snapshot.json', 'r') as f:
    docs = [json.loads(line) for line in f]
    
    
tdf = pd.DataFrame(docs)

# tdf = docs.to_dataframe()
t_df=tdf.drop(['submitter', 'title', 'comments', 'journal-ref', 'doi','report-no', 'license'], axis=1)
# ttt_df=tdf.compute()
# t_df=ttt_df.copy()


time=lambda x: pd.Series(version_year(x['versions']))
t_df[['year','day','month']]=t_df.apply(time,axis=1)
t_df.month=pd.to_datetime(t_df.month, format='%b').dt.month

df = pd.DataFrame({'year': t_df.year, 'month': t_df.month, 'day': t_df.day})

t_df['date']=pd.to_datetime(df)
t_df['author_set']=t_df['authors_parsed'].apply(authors)
t_df['cat']=t_df['categories'].apply(cat)
t_df = t_df.drop(t_df[(t_df.year <2001) | (t_df.year > 2020)].index)
t_df=t_df.sort_values('year')

t_df['cat'].apply(frequency_dict)

freq_group_cat_100={}
for i in freq_group_cat:
    if freq_group_cat[i]>=100:
        freq_group_cat_100[i]=freq_group_cat[i]
        
        
freq_group_cat_n100={}
for i in freq_group_cat:
    if freq_group_cat[i]<100:
        freq_group_cat_n100[i]=freq_group_cat[i]
        
        
t_df[['categories','author_set']].apply(cat_auth,axis=1)
for i in cat_auth_dict:
    t=cat_auth_dict[i]
    #print(i,t)
    cat_auth_dict[i]=set(frozenset(j)for j in t)
    
    
cat_auth_lst=[]
for i in cat_auth_dict:
    cat_auth_lst.append([i,cat_auth_dict[i]])
# len(cat_auth_lst),cat_auth_lst[0]


jacard ={}
for i in range(len(cat_auth_lst)):
    for j in range(i+1,len(cat_auth_lst)):
        jacard[(cat_auth_lst[i][0],cat_auth_lst[j][0])]=len(cat_auth_lst[i][1].intersection(cat_auth_lst[j][1])) / len(cat_auth_lst[i][1].union(cat_auth_lst[j][1]))
        
        
        
c=0
for i in jacard:
    if jacard[i]>=0.01:
        c+=1
        
        
        
t_df['clean_abstract']=t_df['abstract'].map(abs_clean)
t_df['abstract_uncased']=t_df['abstract'].map(abs_uncased)

data = pd.DataFrame(list(t_df.abstract_uncased), columns=['text'])

#data['text'] = t_df.clean_abstracts#.apply(lambda x: x.lower())
data['year'] = list(t_df.date.dt.year)
data['month'] = list(t_df.date.dt.month)
data['day'] = list(t_df.date.dt.day)####
data['user']=list(t_df['cat'].apply(templ))
data['author']=list(t_df.author_set)
data['date']=list(t_df.date)
#data = data.explode('category').dropna(subset=['category']).reset_index(drop=True)
data.drop_duplicates(subset=['text'], inplace=True)
# data.isna().any(),t_df.isna().any()


data = data.rename(columns={'date': 'time'})

N=1000000
train_dev, test = train_test_split(data, test_size=0.2, random_state=123, stratify=data[['year']])
train, dev = train_test_split(train_dev, test_size=0.125, random_state=123, stratify=train_dev[['year']])


train=train.sort_values('time')
test=test.sort_values('time')
dev=dev.sort_values('time')

train.to_csv('arxiv_train.csv', index=False)
dev.to_csv('arxiv_dev.csv', index=False)
test.to_csv('arxiv_test.csv', index=False)

edge_set = set()

users = set()

for i in data.user:
    users.update(i)

tt=[i for i in jacard if jacard[i]>0.01]
edge_set.update(tt)

with open('arxiv_edges.p', 'wb') as f:
    pickle.dump(edge_set, f)
with open('arxiv_users.p', 'wb') as f:
    pickle.dump(users, f)
