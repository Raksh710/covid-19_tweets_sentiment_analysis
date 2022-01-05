#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy


# In[2]:


df = pd.read_csv("Corona_NLP_train.csv")


# In[3]:


df.head()


# In[4]:


df_tweets=df[['OriginalTweet','Sentiment']]


# In[5]:


df_tweets.head()


# In[26]:


100 * df_tweets['Sentiment'].value_counts()/len(df_tweets)


# In[7]:


df_tweets.isna().sum()


# In[8]:


# Checking if there are any "blank" tweets. If there are, then we'll remove such tweets.
blanks = []
for i, tweet, lb in df_tweets.itertuples():
    if type(tweet) == str:
        if tweet.isspace():
            blanks.append(i)


# In[9]:


blanks # no blank tweets


# # EDA

# In[82]:


plt.figure(figsize=(12,6))
sns.histplot(data=df_tweets,x='Sentiment')


# #### Inference:
# We can see that most of the samples (tweets) have "positive" sentiment. Next, are the tweets having "Negative" sentiments, after that comes tweets with "Neutral" sentiments. Then, we can see the "extreme cases" lead by "Extremely Positive" sentiments and followed by "Extremely Negative" sentiments.

# #### Let's try to analyze these tweets with respect to their sentiment polarity scores (positve score, negative score, neutral score and compound score)

# In[69]:


df2 = df_tweets.copy()


# In[70]:


from nltk.sentiment import SentimentIntensityAnalyzer


# In[71]:


sia = SentimentIntensityAnalyzer()


# In[72]:


sia.polarity_scores(df2['OriginalTweet'][3])


# In[73]:


df2['pos_score'] = df2["OriginalTweet"].apply(lambda x: sia.polarity_scores(x)['pos'])
df2['neg_score'] = df2["OriginalTweet"].apply(lambda x: sia.polarity_scores(x)['neg'])
df2['neu_score'] = df2["OriginalTweet"].apply(lambda x: sia.polarity_scores(x)['neu'])
df2['comp_score'] = df2["OriginalTweet"].apply(lambda x: sia.polarity_scores(x)['compound'])


# In[74]:


df2.head()


# In[75]:


plt.figure(figsize=(11,6))
sns.histplot(data=df2,x='comp_score',kde=True)


# #### Inference:
# We can notice almost a perfect normal distribution with wide tails. The plot is almost symmetric, with a slightly highert height around the positive region which could be due to the higher number of positive and extremely positive number of tweets.

# In[78]:


plt.figure(figsize=(11,6))
sns.histplot(data=df2,x='pos_score',kde=True)


# #### Inference:
# The positive score of tweets seems to distributed in a highly right skewed normal distribution manner, with a slight peak occuring around 0.06 - 0.08. However, we can see that a very high number of tweets (around 16000) have a pos_score of 0 indicating that they are either highly negative, negative or neutral. All the non-positive number of tweets are summed in that one single bar.

# In[77]:


plt.figure(figsize=(11,6))
sns.histplot(data=df2,x='neg_score',kde=True)


# #### Inference:
# The negative score of tweets seems to distributed in a highly right skewed normal distribution manner, with a slight peak occuring around 0.08 - 0.12. However, we can see that a very high number of tweets (around 17500) have a neg_score of 0 indicating that they are either highly positive, positive or neutral. All the non-negative number of tweets are summed in that one single bar.

# In[79]:


plt.figure(figsize=(11,6))
sns.histplot(data=df2,x='neu_score',kde=True)


# #### Inference:
# The distribution is of the form highly left skewed normal distribution. We can see "humped peak" around a neu_score of 0.85. High number of "absolutely neutral" tweets are present (around 7000), indicated by a single bar.

# The above 4 graphs show that except comp_score all the scores have a high skewed distribution followed/preceeded by a single long bar (indicating all those scores which are not present in that particular score category)

# #### Now, Let's focus on the comp_score of each sentiment

# #### Neutral

# In[92]:


plt.figure(figsize=(11,9))
sns.histplot(data=df2[df2['Sentiment']=='Neutral'],x='comp_score',kde=True)


# #### Inference:
# For "Neutral" tweets, we can see a peak around comp_score of 0, with a symmetric tails. The distribution is normal and symmetric with long tails. However, the tail on the positve side is slightly longer than the one on the left (due to higher number of positve tweets)

# #### Positive

# In[80]:


plt.figure(figsize=(11,9))
sns.histplot(data=df2[df2['Sentiment']=='Positive'],x='comp_score',kde=True)


# #### Inference:
# For positve tweets, we can see a huge peak in comp_score from around 0.5 and the peak diminishes around 0.75. This could be due to the fact that tweets having comp_score of more than 0.5 are classified as "Highly Positive" and not just "Positive". However, we could see quite a few tweets which have comp_score of more than 0.75 being classified as only "Positive" and not "Highly Positive", this could be due to the fact that those tweets might be having a very high neu_score or neg_score.

# #### Extremely Positive

# In[84]:


plt.figure(figsize=(11,9))
sns.histplot(data=df2[df2['Sentiment']=='Extremely Positive'],x='comp_score',kde=True)


# In[95]:


df2[(df2['Sentiment']=='Extremely Positive') & (df2['comp_score'] < -0.75)]


# #### Inference:
# In case of "Highly Positive" tweets, we can see a huge peak jump around 0.75 which diminshes at 1 (because 1 is the max score). This peak takes place right off from the point where the peak of "Positive" tweets end, proving our assumption (comp_score greater than 0.75 being classified as "Highly Positive" tweets) to be correct. We can even spot a few outliers having the comp_score of around -0.75, which could be due to the fact those tweets having high neg_score and neu_score than the pos_score. It could also be the case such tweets are misclassified (possibly due to some error in the initial classifying system)

# #### Negative

# In[81]:


plt.figure(figsize=(11,9))
sns.histplot(data=df2[df2['Sentiment']=='Negative'],x='comp_score',kde=True)


# #### Inference:
# We can see that the peak of this graph is wobbly and is on the negative side. There is no "one peak", however, we can say that the region of peak starts from around -0.25 and ends a little bit before -0.75. This could be due to the fact that the statements having comp_score lower than -0.75 are classified as "Highly Negative" instead of just "Negative"

# #### Extremely Negative

# In[83]:


plt.figure(figsize=(11,9))
sns.histplot(data=df2[df2['Sentiment']=='Extremely Negative'],x='comp_score',kde=True)


# In[96]:


df2[(df2['Sentiment']=='Extremely Negative') & (df2['comp_score'] > 0.75)]


# #### Inference:
# In case of "Highly Negative" tweets, we can see a huge peak jump around -0.75 which diminshes at 1 (because -1 is the min score). This peak takes place right off from the point where the peak of "Negative" tweets end, proving our assumption (comp_score less than -0.75 being classified as "Highly Negative" tweets) to be correct. We can even spot a few outliers having the comp_score of around +0.75, which could be due to the fact those tweets having high pos_score and neu_score than the pos_score. It could also be the case such tweets are misclassified (possibly due to some error in the initial classifying system)

# #### Note:
# We can notice that the graphs of "Highly Positive" and "Highly Negative" graphs are almost mirror image of each other, indicating an almost symmetric distribution around the "Extreme" values. This was also noticed earlier when we plotted "neu_score" 

# ## Implementing ML algos

# #### Splitting the data into train and test set

# In[11]:


X = df_tweets['OriginalTweet']
y = df_tweets['Sentiment']


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# #### Creating pipelines (including Tfidf vectorization and then using classification algorithm)

# In[14]:


from sklearn.pipeline import Pipeline


# In[10]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[15]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.naive_bayes import MultinomialNB


# In[42]:


p1 = Pipeline([('tfidf',TfidfVectorizer()), ('lr',LogisticRegression(max_iter=1000000,solver='saga'))]) # Logistic regression pipeline
p2 =  Pipeline([('tfidf',TfidfVectorizer()), ('lsvc',LinearSVC(random_state=42))]) # Linear SVC regression pipeline
p3 = Pipeline([('tfidf',TfidfVectorizer()), ('svc',SVC(random_state=42))]) # SVC regression pipeline
p4 =  Pipeline([('tfidf',TfidfVectorizer()), ('mnb',MultinomialNB())]) # Multinomial Naive bayes regression pipeline


# #### 1) Logistic Regression

# In[39]:


p1.fit(X_train,y_train)


# In[40]:


lr_pred = p1.predict(X_test)


# In[21]:


from sklearn.metrics import classification_report,confusion_matrix


# In[41]:


print(classification_report(y_test,lr_pred))
print(confusion_matrix(y_test,lr_pred))


# #### 2) Linear SVC

# In[24]:


p2.fit(X_train,y_train)
lsvc_pred = p2.predict(X_test)
print(classification_report(y_test,lsvc_pred))
print(confusion_matrix(y_test,lsvc_pred))


# #### 3) SVC

# In[43]:


# SVC not good for this case
p3.fit(X_train,y_train)
svc_pred = p3.predict(X_test)
print(classification_report(y_test,svc_pred))
print(confusion_matrix(y_test,svc_pred))


# #### 4) Multinomial NB

# In[28]:


p4.fit(X_train,y_train)
mnb_pred = p4.predict(X_test)
print(classification_report(y_test,mnb_pred))
print(confusion_matrix(y_test,mnb_pred))


# #### 5) Random Forest

# In[29]:


from sklearn.ensemble import RandomForestClassifier


# In[30]:


p5 = Pipeline([('tfidf',TfidfVectorizer()), ('rf',RandomForestClassifier(n_estimators=100,random_state=42))])


# In[31]:


p5.fit(X_train,y_train)
rf_pred = p5.predict(X_test)
print(classification_report(y_test,rf_pred))
print(confusion_matrix(y_test,rf_pred))


# #### 6) XGBOOST

# In[44]:


from xgboost import XGBClassifier


# In[45]:


p6 = Pipeline([('tfidf',TfidfVectorizer()), ("xgb",XGBClassifier(random_state=42,booster='dart'))])


# In[46]:


p6.fit(X_train,y_train)
xgb_pred = p6.predict(X_test)
print(classification_report(y_test,xgb_pred))
print(confusion_matrix(y_test,xgb_pred))


# #### 7) XGBRFBOOST

# In[47]:


from xgboost import XGBRFClassifier


# In[49]:


p7 = Pipeline([('tfidf',TfidfVectorizer()), ("xgbrf",XGBRFClassifier(random_state=42,booster='dart'))])
p7.fit(X_train,y_train)
xgbrf_pred = p7.predict(X_test)
print(classification_report(y_test,xgbrf_pred))
print(confusion_matrix(y_test,xgbrf_pred))


# #### 8) Catboost

# In[50]:


from catboost import CatBoostClassifier


# In[53]:


tfidf = TfidfVectorizer()


# In[54]:


X_train_vect = tfidf.fit_transform(X_train)
X_test_vect = tfidf.transform(X_test)


# In[58]:


p8 = Pipeline([('tfidf',TfidfVectorizer()), ("cb",CatBoostClassifier(random_state=42))])
p8.fit(X_train,y_train)
cb_pred = p8.predict(X_test)
print(classification_report(y_test,cb_pred))
print(confusion_matrix(y_test,cb_pred))


# ### Catboost is giving us the best accuracy... let's try it on the actual test set

# #### Note: We won't be doing EDA on this actual test data as in practical scenario it is not possible to do any EDA on the incoming test data 

# In[59]:


df_test = pd.read_csv("Corona_NLP_test.csv")


# In[60]:


df_test.head()


# In[66]:


100 * df_test['Sentiment'].value_counts()/len(df_test)


# In[62]:


X_final = df_test['OriginalTweet']
y_final = df_test['Sentiment']


# In[63]:


cb_test_pred = p8.predict(X_final)


# In[64]:


print(classification_report(y_final,cb_test_pred))
print(confusion_matrix(y_final,cb_test_pred))


# ## Catboost gave us an accuracy of around 57% on the actual test data as well

# In[ ]:




