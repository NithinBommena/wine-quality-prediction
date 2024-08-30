#!/usr/bin/env python
# coding: utf-8

# # Import The Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score


# # Import The Dataset

# In[2]:


df = pd.read_csv('winequality.csv')
df.head()


# In[3]:


df.describe()


# In[4]:


df.info()


# # Taking Care Of Missing Data

# In[5]:


df.isnull().sum()


# In[6]:


for col, value in df.items():
    if col != 'type':
        df[col] = df[col].fillna(df[col].mean())


# In[7]:


df.isnull().sum()


# In[8]:


df.corr()


# # Data Visualization

# In[9]:


sns.pairplot(df)


# In[10]:


sns.catplot(x='quality', data = df, kind = 'count')


# In[11]:


sns.countplot(df['type'])


# In[12]:


plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y = 'volatile acidity', data = df)


# In[13]:


plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y = 'citric acid', data = df)


# # Correlation Matrix

# In[14]:


corr = df.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr, annot=True, cmap='coolwarm')


# # Splitting Dataset Into Dependent And Independent Variables And Labelling

# In[15]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()


# In[16]:


df['type']=lb.fit_transform(df['type'])
df


# white-1,red-0

# In[17]:


x=df.drop('quality',axis=1)
x


# In[18]:


y=df['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)
y


# # Splitting Data Into Train And Test

# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)


# In[21]:


y.shape, y_train.shape, y_test.shape


# # Logistic Regression

# In[22]:


from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler #Feature scaling
from sklearn.pipeline import Pipeline #pipeline
model1=Pipeline([('rescale',StandardScaler()),('classifier',LogisticRegression())])
model1.fit(x_train,y_train)


# In[23]:


x_test_prediction = model1.predict(x_test)
acc1= accuracy_score(x_test_prediction, y_test)
acc1


# # SVM

# In[24]:


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler#Feature scaling
from sklearn.pipeline import Pipeline #pipeline
model2=Pipeline([('rescale',StandardScaler()),('classifier',SVC(kernel='rbf'))])
model2.fit(x_train,y_train)


# In[25]:


x_test_prediction = model2.predict(x_test)
acc2= accuracy_score(x_test_prediction, y_test)
acc2


# # KNN

# In[26]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler#Feature scaling
from sklearn.pipeline import Pipeline # Pipeline
model3 = Pipeline([('rescale', StandardScaler()),('classifier', KNeighborsClassifier())])
model3.fit(x_train,y_train)


# In[27]:


x_test_prediction = model3.predict(x_test)
acc3= accuracy_score(x_test_prediction, y_test)
acc3


# # Decision Tree

# In[28]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler#Feature scaling
from sklearn.pipeline import Pipeline # Pipeline
model4 = Pipeline([('rescale', StandardScaler()),('classifier', DecisionTreeClassifier())])
model4.fit(x_train,y_train)


# In[29]:


x_test_prediction = model4.predict(x_test)
acc4= accuracy_score(x_test_prediction, y_test)
acc4


# # Random Forest Classifier

# In[30]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler#Feature scaling
from sklearn.pipeline import Pipeline # Pipeline
model5= Pipeline([('rescale', StandardScaler()),('classifier', DecisionTreeClassifier())])
model5.fit(x_train,y_train)


# In[31]:


x_test_prediction = model5.predict(x_test)
acc5= accuracy_score(x_test_prediction, y_test)
acc5


# In[32]:


Accuracy={'logistic':acc1,'SVM':acc2,'KNN':acc3,'Decision Tree':acc4,'Random Forest':acc5}


# In[33]:


plt.ylabel('% Accuracy')
plt.bar(Accuracy.keys(),Accuracy.values())


# In[34]:


max(Accuracy,key=Accuracy.get)


# # Building a Predictive System

# In[38]:


input_data=input().split()
if input_data[0]=='white':
    input_data[0]='1'
else:
    input_data[0]='0'
# changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)
# reshape the data as we are predicting the label for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model5.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==1):
  print('Good Quality Wine')
else:
  print('Bad Quality Wine')


# In[ ]:




