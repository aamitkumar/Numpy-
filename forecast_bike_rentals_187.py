#!/usr/bin/env python
# coding: utf-8

# In[188]:


import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import matplotlib.pyplot as plt
import os
np.random.seed(42)


# In[189]:


filePath="/cxldata/datasets/project/bikes.csv"
bikesData=pd.read_csv(filePath)
bikesData.head(10)


# In[190]:


type(bikesData)


# In[191]:


bikesData.info()


# In[192]:


bikesData.count(axis=0).count()


# In[193]:


bikesData["yr"].unique()


# In[194]:


dir(bikesData)


# In[195]:


bikesData.describe()


# In[196]:


columnsToDrop=["instant","casual","registered","atemp","dteday"]
bikesData=bikesData.drop(columnsToDrop,axis=1)


# In[197]:


bikesData


# In[198]:


np.random.seed(42)


# In[199]:


from sklearn.model_selection import train_test_split


# In[200]:


bikesData['dayCount'] = pd.Series(range(bikesData.shape[0]))/24


# In[201]:


train_set,test_set=train_test_split(bikesData,test_size=0.3,random_state=42)


# In[202]:


train_set.sort_values('dayCount', axis= 0, inplace=True)
test_set.sort_values('dayCount', axis= 0, inplace=True)


# In[203]:


print(train_set)


# In[204]:


print(test_set)


# In[205]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# In[206]:


columnsToScale=['temp','hum','windspeed']
scaler=StandardScaler()


# In[207]:


train_set[columnsToScale] = scaler.fit_transform(train_set[columnsToScale])


# In[208]:


test_set[columnsToScale] = scaler.transform(test_set[columnsToScale])


# In[209]:


bikesData.describe()


# In[210]:


train_set.describe()


# In[211]:


train_set["temp"].mean()+ test_set["temp"].mean()


# In[212]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost
from xgboost import XGBRegressor 


# In[213]:


trainingCols=train_set.drop("cnt",axis=1)
trainingLabels=train_set["cnt"]


# In[214]:


dec_reg=DecisionTreeRegressor(random_state=42)


# In[215]:


dt_mae_scores = -cross_val_score(dec_reg,trainingCols,trainingLabels,scoring="neg_mean_absolute_error",cv=10)
display_scores(dt_mae_scores)


# In[216]:


dt_mse_scores= np.sqrt(-cross_val_score(dec_reg,trainingCols,trainingLabels,scoring="neg_mean_squared_error",cv=10))
display_scores(dt_mse_scores)


# In[217]:


lin_reg=LinearRegression()


# In[218]:


lr_mae_scores = -cross_val_score(lin_reg,trainingCols,trainingLabels,cv=10,scoring="neg_mean_absolute_error")
display_scores(lr_mae_scores)


# In[219]:


lr_mse_scores = np.sqrt(-cross_val_score(lin_reg,trainingCols,trainingLabels,cv=10,scoring="neg_mean_squared_error"))
display_scores(lr_mse_scores)


# In[220]:


forest_reg=RandomForestRegressor(random_state=42,n_estimators=150)


# In[221]:


rf_mae_scores = -cross_val_score(forest_reg,trainingCols,trainingLabels,cv=10,scoring="neg_mean_absolute_error")
display_scores(rf_mae_scores)


# In[222]:


rf_mse_scores = np.sqrt(-cross_val_score(forest_reg,trainingCols,trainingLabels,cv=10,scoring="neg_mean_squared_error"))
display_scores(rf_mse_scores)


# In[223]:


from sklearn.model_selection import GridSearchCV 


# In[224]:


param_grid = [
    {'n_estimators': [120, 150], 'max_features': [10, 12], 'max_depth': [15, 28]},
]


# In[225]:


forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,scoring='neg_mean_squared_error')


# In[226]:


grid_search.fit(trainingCols,trainingLabels)


# In[227]:


grid_search.best_params_


# In[228]:


feature_importances = grid_search.best_estimator_.feature_importances_


# In[229]:


print(feature_importances)


# In[236]:


final_model = grid_search.best_estimator_
test_set.sort_values('dayCount', axis= 0, inplace=True)
test_x_cols = (test_set.drop(['cnt'], axis=1)).columns.values
test_y_cols = 'cnt'

X_test = test_set.loc[:,test_x_cols]
y_test = test_set.loc[:,test_y_cols]


# In[244]:


test_set.loc[:,'predictedCounts_test'] = final_model.predict(X_test)

mse = mean_squared_error(y_test, test_set.loc[:,'predictedCounts_test'])
final_mse = np.sqrt(mse)
print(final_mse)
test_set.describe()


# In[ ]:




