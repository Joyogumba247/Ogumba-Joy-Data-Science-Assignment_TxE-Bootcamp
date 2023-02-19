#!/usr/bin/env python
# coding: utf-8

# # My Name Is JOY OGUMBA

# # This is my Assignment

# ### Step 1: Importing Libraries

# In[2]:


import pandas as pd

df = pd.read_csv(r"C:\Users\HP PAVILION\Downloads\titanic.csv")

print(df)


# In[3]:


print(df.head(10))


# In[4]:


print(df.tail(10))


# In[10]:


### Reading the headers
print(df.columns)


# In[17]:


### Read each column
print(df['PassengerId'][0:-1])


# In[18]:


### Read each row
print(df.iloc[1])


# In[21]:


### Read a specific location 
print(df.iloc[5,3])


# In[24]:


df.loc[df['Cabin'] == "Nan"]


# In[25]:


df.describe()


# In[32]:


## Sorting the values
df.sort_values ('PassengerId', ascending=True)


# In[33]:


### Data Dimension
df.shape


# In[1]:


import numpy as np
import pandas as pd


# In[9]:


from pandas_profiling import ProfileReport


# In[7]:



import pandas as pd

df = pd.read_csv(r"C:\Users\HP PAVILION\Downloads\titanic.csv")


# In[10]:


print(df)


# In[11]:


profile = ProfileReport(df)


# In[12]:


print(profile)


# In[5]:


import pandas as pd
df = pd.read_csv(r"C:\Users\HP PAVILION\Downloads\titanic.csv")

print(df)


# In[6]:


### Check for missing values

df.isnull().sum()


# In[7]:


df= df.fillna(0)


# In[8]:


### Replace missing value with zero

df.isnull().sum()


# In[22]:


df.to_csv('titanic.csv', index=False)


# In[21]:


get_ipython().system(' ls')


# In[24]:


## Exploratory Data Analysis
### Read Data
df = pd.read_csv('titanic.csv')


# In[25]:


df


# In[27]:


pd.set_option('display.max_rows', df.shape[0]+1)


# In[28]:


df


# In[29]:


pd.set_option('display.max_rows', 10)


# In[30]:


df


# In[32]:


### Overview of data types of each columns in the dataframe

df.dtypes


# In[33]:


### Showing specific data types
df.select_dtypes(include=['number'])


# In[34]:


df.select_dtypes(include=['object'])


# # Questions
# 
# ## How many passengers are in the dataset?

# In[35]:


import pandas as pd

df = pd.read_csv('titanic.csv')
num_passengers = len(df)
print(f'The number of passengers in the dataset is {num_passengers}.')


# ## How many survived, how many did not?

# In[37]:


import pandas as pd

df = pd.read_csv('titanic.csv')
survival_counts = df['Survived'].value_counts()
num_survived = survival_counts[1]
num_not_survived = survival_counts[0]
print(f'{num_survived} passengers survived and {num_not_survived} did not.')


# ## What is the distribution of passenger classes?

# In[38]:


import pandas as pd

df = pd.read_csv('titanic.csv')
pclass_counts = df['Pclass'].value_counts()
print('Passenger class distribution:')
for pclass, count in pclass_counts.items():
    print(f'Class {pclass}: {count} passengers')


# ## How many passengers are male and how many are female?

# In[39]:


import pandas as pd

df = pd.read_csv('titanic.csv')
sex_counts = df['Sex'].value_counts()
num_male = sex_counts['male']
num_female = sex_counts['female']
print(f'There are {num_male} male passengers and {num_female} female passengers.')


# ## What is the distribution of passenger ages? Are there any outliers or missing values?

# In[40]:


import pandas as pd

df = pd.read_csv('titanic.csv')
age_counts = df['Age'].value_counts(dropna=False)
print(f'Age distribution:')
for age, count in age_counts.items():
    if pd.isna(age):
        print(f'Missing: {count} passengers')
    else:
        print(f'Age {age}: {count} passengers')


# ## How many siblings/spouses (SibSp) and parents/children (Parch) did the passengers have?

# In[41]:


import pandas as pd

df = pd.read_csv('titanic.csv')
sibsp_counts = df['SibSp'].value_counts()
parch_counts = df['Parch'].value_counts()
print('Siblings/spouses:')
for sibsp, count in sibsp_counts.items():
    print(f'{count} passengers had {sibsp} siblings/spouses')
print('Parents/children:')
for parch, count in parch_counts.items():
    print(f'{count} passengers had {parch} parents/children')


# ## What was the fare distribution and were there any outliers?

# In[42]:


import pandas as pd

df = pd.read_csv('titanic.csv')
fare_counts = df['Fare'].value_counts(bins=10, sort=False)
print(f'Fare distribution:')
print(fare_counts)


# ## What was the distribution of embarkation points (Embarked)?

# In[43]:


import pandas as pd

df = pd.read_csv('titanic.csv')
embarked_counts = df['Embarked'].value_counts()
print('Embarkation point distribution:')
for embark, count in embarked_counts.items():
    print(f'{count} passengers embarked at {embark}')


# ## Is there a correlation between passenger class and survival rate?

# In[44]:


import pandas as pd

df = pd.read_csv('titanic.csv')
survival_by_class = df.groupby('Pclass')['Survived'].mean()
print('Survival rate by passenger class:')
print(survival_by_class)


# # Pre-processing the dataset to get ready for ML Application

# In[46]:


import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


# In[48]:


### Load the data set

titanic = pd.read_csv('titanic.csv')


# In[49]:


### Drop unnecessary columns

titanic = titanic.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)


# In[50]:


### Handle missing values in the 'Age' and 'Embarked' columns

titanic['Age'] = titanic['Age'].fillna(titanic['Age'].mean())
titanic = titanic.dropna(subset=['Embarked'])


# In[51]:


### # Convert categorical data to numerical data

titanic_encoded = pd.get_dummies(titanic, columns=['Sex', 'Embarked'])


# In[52]:


### Scale the numerical data

scaler = StandardScaler()
titanic_encoded[['Age', 'Fare']] = scaler.fit_transform(titanic_encoded[['Age', 'Fare']])


# In[53]:


### Split the dataset into training and testing set

X = titanic_encoded.drop(['Survived'], axis=1)
y = titanic_encoded['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


### Using Logistic Regression model


# In[54]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict the test data and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)


# # Conclusion
# 
# ### In summary, an accuracy of 0.7877 is a decent starting point, but further analysis and experimentation is required to determine if the model is good enough for the intended use case. Hence, it is ready for ML and further  processing for a more confirmations.
# 
# ### Thank you

# In[ ]:





# In[ ]:




