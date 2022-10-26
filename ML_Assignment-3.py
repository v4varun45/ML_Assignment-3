#!/usr/bin/env python
# coding: utf-8

# # Titanic Dataset
# Q1)
# In[2]:


import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import*
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore")


# In[3]:


Data=pd.read_csv('titanic.csv')


# In[4]:


Data.head()


# In[5]:


Data.tail()


# In[6]:


Data.describe()


# In[7]:


new_age=np.where(Data['Age'].isnull(),29,Data['Age'])
Data['Age']=new_age


# In[8]:


encoder=preprocessing.LabelEncoder()
Data['Sex']=encoder.fit_transform(Data['Sex'])


# In[9]:


Data.isnull().sum()


# In[10]:


Data['Sex'].corr(Data['Survived'])

a) Yes
# In[ ]:




# Q2) Visualizations
# In[11]:


plt.style.use('seaborn')


# In[12]:


a=Data['Sex']
b=Data['Survived']


# In[13]:


plt.scatter(a, b)


# In[14]:


plt.plot(a, b)

# Naive Bayes
# In[15]:


y=Data['Survived']
X=Data.drop(['Survived','Name','Ticket','Cabin','Embarked'],axis=1)


# In[16]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# In[17]:


clf=BernoulliNB()


# In[18]:


clf.fit(X_train,y_train)


# In[19]:


y_pred=clf.predict(X_test)


# In[20]:


accuracy_score(y_test,y_pred,normalize=True)


# In[ ]:





# In[21]:


Data.dtypes


# In[ ]:





# In[ ]:





# # Glass Dataset
# Naive Bayes
# In[22]:


Csv_dataset=pd.read_csv("glass.csv")


# In[23]:


Csv_dataset.isnull().sum()


# In[24]:


Csv_dataset.head()


# In[25]:


Csv_dataset.tail()


# In[26]:


Csv_dataset.describe()


# In[27]:


features = ['Rl', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
target = 'Type'

X_train, X_test, y_train, y_test = train_test_split(Csv_dataset[::-1], Csv_dataset['Type'],test_size=0.2, random_state=1)

classifier = GaussianNB()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


# In[28]:


print(classification_report(y_test, y_pred))


# In[29]:


print(confusion_matrix(y_test, y_pred))


# In[30]:


print('accuracy is',accuracy_score(y_test, y_pred))


# In[ ]:




# SVM
# In[31]:


from sklearn.svm import SVC
classifier=SVC()


# In[32]:


from sklearn.svm import SVC, LinearSVC

classifier = LinearSVC()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


# In[33]:


print(classification_report(y_test, y_pred))


# In[34]:


print(confusion_matrix(y_test, y_pred))


# In[35]:


print('accuracy is',accuracy_score(y_test, y_pred))


# In[36]:


visual = Csv_dataset.corr()


# In[37]:


Csv_dataset.corr().style.background_gradient(cmap="Greens")


# In[38]:


sns.heatmap(visual, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
plt.show()


# Naive Bayes model did very good job. Got Accuracy score of 83.72
