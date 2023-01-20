#!/usr/bin/env python
# coding: utf-8

# # <font color='Purple' style="font-family:Castellar"> **BDM 1034 - Application Design for Big Data 02 (DSMM Group 2)** </font> 

# In[3]:


from IPython import display
display.Image("https://static.startuptalky.com/2021/12/Netflix-Recommendation-Engine-Working-StartupTalky.jpg")


# # <font color='Teal' style="font-family:constantia"> *FINAL PROJECT - Group A* </font>

# ## <font color='Teal' style="font-family:constantia"> Members:
# >**Gene Martin Gamboa**
#     
# >**Ila Sharma**
#     
# >**Francisco Villa Saavedra**
# </font>

# ## <font color='Teal' style="font-family:constantia"> Objectives: </font>

# #### This project aims to build a movie recommendation mechanism within Netflix. We plan to use numpy, pandas, seaborn and other basic tools we've learn from this course in order to complete this project. The data we've selected to use have more than 8,000 records where we will base our analysis from. The first step in our approach is to perform Exploratory Data Analysis (EDA) on the data. Then we will build/code the recommendation system then followed by getting the recommendations from the user.

# In[4]:


pip install plotly


# In[5]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt


# ## <font color='Orange' style="font-family:constantia"> Perform Exploratory Data Analysis (EDA) on the data: </font>

# In[6]:


netflix_overall=pd.read_csv("netflix_titles.csv")
netflix_overall.head()


# In[7]:


netflix_overall.info()


# In[8]:


netflix_overall.count()


# In[9]:


netflix_shows=netflix_overall[netflix_overall['type']=='TV Show']


# In[10]:


netflix_shows


# In[11]:


netflix_movies=netflix_overall[netflix_overall['type']=='Movie']


# In[12]:


netflix_movies


# In[13]:


sns.set(style="darkgrid")
ax = sns.countplot(x="type", data=netflix_overall, palette="Set2")


# In[14]:


## Movie rating analysis

plt.figure(figsize=(12,10))
sns.set(style="darkgrid")
ax = sns.countplot(x="rating", data=netflix_movies, palette="Set2", order=netflix_movies['rating'].value_counts().index[0:15])


# In[15]:


## Year wise analysis

plt.figure(figsize=(12,10))
sns.set(style="darkgrid")
ax = sns.countplot(y="release_year", data=netflix_movies, palette="Set2", order=netflix_movies['release_year'].value_counts().index[0:15])


# In[16]:


countries={}
netflix_movies['country']=netflix_movies['country'].fillna('Unknown')
cou=list(netflix_movies['country'])
for i in cou:
    #print(i)
    i=list(i.split(','))
    if len(i)==1:
        if i in list(countries.keys()):
            countries[i]+=1
        else:
            countries[i[0]]=1
    else:
        for j in i:
            if j in list(countries.keys()):
                countries[j]+=1
            else:
                countries[j]=1


# In[17]:


countries_fin={}
for country,no in countries.items():
    country=country.replace(' ','')
    if country in list(countries_fin.keys()):
        countries_fin[country]+=no
    else:
        countries_fin[country]=no
        
countries_fin={k: v for k, v in sorted(countries_fin.items(), key=lambda item: item[1], reverse= True)}


# In[18]:


## Top 10 movie content creating countries

plt.figure(figsize=(8,8))
ax = sns.barplot(x=list(countries_fin.keys())[0:10],y=list(countries_fin.values())[0:10])
ax.set_xticklabels(list(countries_fin.keys())[0:10],rotation = 90)


# In[19]:


from collections import Counter

genres=list(netflix_movies['listed_in'])
gen=[]

for i in genres:
    i=list(i.split(','))
    for j in i:
        gen.append(j.replace(' ',""))
g=Counter(gen)


# In[20]:


## Plotting genre of movies vs their count

g={k: v for k, v in sorted(g.items(), key=lambda item: item[1], reverse= True)}


fig, ax = plt.subplots()

fig = plt.figure(figsize = (10, 10))
x=list(g.keys())
y=list(g.values())
ax.vlines(x, ymin=0, ymax=y, color='green')
ax.plot(x,y, "o", color='maroon')
ax.set_xticklabels(x, rotation = 90)
ax.set_ylabel("Count of movies")
# set a title
ax.set_title("Genres");


# In[21]:


## Analysis of TV series in Netflix
countries1={}
netflix_shows['country']=netflix_shows['country'].fillna('Unknown')
cou1=list(netflix_shows['country'])
for i in cou1:
    #print(i)
    i=list(i.split(','))
    if len(i)==1:
        if i in list(countries1.keys()):
            countries1[i]+=1
        else:
            countries1[i[0]]=1
    else:
        for j in i:
            if j in list(countries1.keys()):
                countries1[j]+=1
            else:
                countries1[j]=1


# In[22]:


countries_fin1={}
for country,no in countries1.items():
    country=country.replace(' ','')
    if country in list(countries_fin1.keys()):
        countries_fin1[country]+=no
    else:
        countries_fin1[country]=no
        
countries_fin1={k: v for k, v in sorted(countries_fin1.items(), key=lambda item: item[1], reverse= True)}


# In[23]:


# Set the width and height of the figure
plt.figure(figsize=(15,15))

# Add title
plt.title("Content creating countries")

# Bar chart showing average arrival delay for Spirit Airlines flights by month
sns.barplot(y=list(countries_fin1.keys()), x=list(countries_fin1.values()))

# Add label for vertical axis
plt.ylabel("Arrival delay (in minutes)")


# In[24]:


features=['title','duration']
durations= netflix_shows[features]

durations['no_of_seasons']=durations['duration'].str.replace(' Season','')

#durations['no_of_seasons']=durations['no_of_seasons'].astype(str).astype(int)
durations['no_of_seasons']=durations['no_of_seasons'].str.replace('s','')


# In[25]:


durations['no_of_seasons']=durations['no_of_seasons'].astype(str).astype(int)


# In[26]:


## TV SHOWS with largest number of seasons

t=['title','no_of_seasons']
top=durations[t]

top=top.sort_values(by='no_of_seasons', ascending=False)


# In[27]:


top20=top[0:20]
top20.plot(kind='bar',x='title',y='no_of_seasons', color='red')


# In[28]:


## lowest number of seasons

bottom=top.sort_values(by='no_of_seasons')
bottom=bottom[20:50]

import plotly.graph_objects as go

fig = go.Figure(data=[go.Table(header=dict(values=['Title', 'No of seasons']),
                 cells=dict(values=[bottom['title'],bottom['no_of_seasons']],fill_color='lavender'))
                     ])
fig.show()


# ## <font color='Orange' style="font-family:constantia"> Build the recommendation system: </font>

# In[29]:


from sklearn.feature_extraction.text import TfidfVectorizer

#removing stopwords
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
netflix_overall['description'] = netflix_overall['description'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(netflix_overall['description'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape


# In[30]:


# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[31]:


indices = pd.Series(netflix_overall.index, index=netflix_overall['title']).drop_duplicates()


# In[32]:


def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return netflix_overall['title'].iloc[movie_indices]


# In[33]:


get_recommendations('Peaky Blinders')


# In[34]:


get_recommendations('Mortel')


# '''Content based filtering on the following factors:
# 
# 1. Title
# 
# 2. Cast
# 
# 3. Director
# 
# 4. Listed in
# 
# 5. Plot'''

# In[35]:


filledna=netflix_overall.fillna('')
filledna.head(2)


# In[36]:


## cleaning data and making all words to lowercase

def clean_data(x):
        return str.lower(x.replace(" ", ""))


# In[37]:


## identifying features on which model is to be filtered

features=['title','director','cast','listed_in','description']
filledna=filledna[features]


# In[38]:


for feature in features:
    filledna[feature] = filledna[feature].apply(clean_data)
    
filledna.head(2)


# In[39]:


## Create BOW(BAG OF WORDS) for all rows

def create_soup(x):
    return x['title']+ ' ' + x['director'] + ' ' + x['cast'] + ' ' +x['listed_in']+' '+ x['description']


filledna['soup'] = filledna.apply(create_soup, axis=1)


# In[40]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(filledna['soup'])

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)


# In[41]:



filledna=filledna.reset_index()
indices = pd.Series(filledna.index, index=filledna['title'])


# ## <font color='Orange' style="font-family:constantia"> Get recommendations: </font>

# In[42]:


def get_recommendations_new(title, cosine_sim=cosine_sim):
    title=title.replace(' ','').lower()
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return netflix_overall['title'].iloc[movie_indices]


# In[43]:


get_recommendations_new('PK', cosine_sim2)


# In[44]:


get_recommendations_new('Peaky Blinders', cosine_sim2)


# In[45]:


get_recommendations_new('The Hook Up Plan', cosine_sim2)


# In[48]:


get_recommendations_new('The Walking Dead', cosine_sim2)


# In[ ]:




