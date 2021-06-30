#!/usr/bin/env python
# coding: utf-8

# # Insurance Data Analysis

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[2]:


df = pd.read_csv (r'C:\Users\dp\Downloads\InsuranceData\insurance.csv')
pd.set_option("display.max_rows", None, "display.max_columns", None)

df1 = df

for i in range(len(df1['smoker'])):
    if df1['smoker'][i] == 'yes':
        df1['smoker'][i] = 1
    else:
        df1['smoker'][i] = 0

df1


# # Visualization

# In[95]:


plt.style.use('seaborn-colorblind')

smokers = [df1['charges'][i] for i in range(len(df1['charges'])) if df1['smoker'][i] == 1]
non_smokers = [df1['charges'][i] for i in range(len(df1['charges'])) if df1['smoker'][i] == 0]

df2 = df1[['smoker','region', 'charges']].copy()
clrs = ['blue' if df2['smoker'][i] == 1 else 'red' for i in range(len(df2['smoker']))]

ax = sns.barplot(x = df2['smoker'], y = df2['charges'], palette = clrs)
ax.set_xticklabels(('Non-Smokers', 'Smokers'))
ax.set_xlabel('')
ax.set_ylabel('Insurance Charges')


# In[111]:


df3 = df1.groupby('age').mean()

plt.style.use('seaborn-colorblind')
df3 = df3.reset_index()

df3.plot.scatter('age', 'charges')

m,c = np.polyfit(df3['age'],df3['charges'], 1)

plt.plot(df3['age'], m*df3['age'] + c, color = 'red')
df3.plot('age', 'charges')


# # Regression Techniques

# In[73]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics.regression import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures


# In[74]:


for i in range(len(df['age'])):
    if df['sex'][i] == 'female':
        df['sex'][i] = 0
    else:
        df['sex'][i] = 1
        
df


# In[75]:


x = np.array(df[df.columns[0:5]])
y = np.array(df['charges'])
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)


# In[78]:


linreg = LinearRegression().fit(X_train, y_train)
pred = linreg.predict(X_test)
r2score = r2_score(y_test, pred)

r2score


# In[119]:


lasso = Lasso(alpha = 0.001, max_iter = 1000).fit(X_train, y_train)
pred = lasso.predict(X_test)
r2score = r2_score(y_test, pred)
r2score


# In[106]:


ridge = Ridge(alpha = 0.0001).fit(X_train, y_train)
pred = ridge.predict(X_test)
r2score = r2_score(y_test, pred)

r2score


# In[113]:


poly = PolynomialFeatures(degree = 2)
X_poly = poly.fit_transform(X_train) 
linreg = LinearRegression().fit(X_poly, y_train)
y_train_pred = linreg.predict(X_poly)
X_test_poly = poly.transform(X_test)
y_test_pred = linreg.predict(X_test_poly)


r2score = r2_score(y_test,y_test_pred)
r2score


# In[ ]:




