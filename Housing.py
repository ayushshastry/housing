#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import time
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from matplotlib import rcParams
import warnings


# # Data Loading

# In[2]:


training = pd.read_csv("/Users/ayushshastry/Desktop/kagglecomp/houseprices/train.csv")
testing = pd.read_csv("/Users/ayushshastry/Desktop/kagglecomp/houseprices/test.csv")


# # Information on Dataset and Cleaning

# In[3]:


testing.info()


# In[4]:


testing.head()


# In[5]:


training.head()


# In[6]:


training.isnull().sum()


# In[7]:


training.shape


# In[8]:


training.info()


# In[9]:


full = pd.concat([training, testing], ignore_index = True)


# In[10]:


full.info()


# In[11]:


full.isnull().sum()


# In[12]:


corr = full.corr(numeric_only = True)
corr


# In[13]:


for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        if abs(corr.iloc[i, j]) >= 0.85:
            print(f"Correlation coefficient {corr.iloc[i, j]:.2f} between {corr.columns[i]} and {corr.columns[j]}")


# In[14]:


full.drop(columns = ['GarageCars', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu'], inplace = True)


# In[15]:


full.info()


# # Data Visualizations

# In[16]:


fig = sns.histplot(x = full['SaleType'], color = "red")


# In[17]:


full['SaleType'].value_counts()


# In[18]:


fig = sns.histplot(x = full['SaleCondition'], color = "red", bins = 25)
fig


# In[19]:


full['SaleCondition'].value_counts()


# In[20]:


fig = px.pie(full, values='SalePrice', names='YrSold', title='Sales by Year')
fig.show()


# In[21]:


fig = sns.histplot(x = full['LotShape'], color = "red")
fig


# In[22]:


fig = sns.histplot(x = full['MSZoning'], color = "red")
fig


# In[23]:


fig = sns.histplot(x = full['Street'], color = "red")
fig


# In[24]:


full['Street'].value_counts()


# In[25]:


fig = sns.histplot(x = full['Utilities'], color = "red")
fig


# In[26]:


full['Utilities'].value_counts()


# In[27]:


plt.figure(figsize=(25,15))
fig = sns.histplot(x = full['Neighborhood'], color = "red", kde = True)
fig


# In[28]:


full['Neighborhood'].value_counts()


# In[29]:


fig = sns.histplot(x = full['BldgType'], color = "red")
fig


# In[30]:


full['BldgType'].value_counts()


# In[31]:


fig = sns.histplot(x = full['HouseStyle'], color = "red")
fig


# In[32]:


full['HouseStyle'].value_counts()


# In[33]:


full.info()


# In[34]:


full.drop(columns = ['Utilities', 'Street'], inplace = True)


# In[35]:


full[['1stFlrSF', '2ndFlrSF', 'SalePrice']]


# In[36]:


fig = px.scatter(full, x = 'GrLivArea', y = 'SalePrice')
fig


# In[37]:


fig = px.scatter(full, x = '1stFlrSF', y = 'SalePrice')
fig


# In[38]:


fig = px.scatter(full, x = '2ndFlrSF', y = 'SalePrice')
fig


# In[39]:


fig = sns.histplot(full['CentralAir'], color = "red")
fig


# In[40]:


full["CentralAir"].value_counts()


# In[41]:


fig = sns.histplot(full['BsmtFullBath'], color = "red")
fig


# In[42]:


full['BsmtFullBath'].value_counts()


# In[43]:


fig = sns.histplot(full['BsmtHalfBath'], color = "red")
fig


# In[44]:


full['BsmtHalfBath'].value_counts()


# In[45]:


fig = sns.histplot(full['KitchenQual'], color = "red")
fig


# In[46]:


full['KitchenQual'].value_counts()


# In[47]:


fig = px.bar(full, x = 'KitchenQual', y = 'SalePrice')
fig.show()


# In[48]:


fig = px.bar(full, x = 'BsmtFullBath', y = 'SalePrice')
fig.show()


# # Dealing With Null Values

# In[49]:


missing = full.isnull().sum()
missing


# In[50]:


missing = missing.sort_values(ascending = False)


# In[51]:


missing.head(28)


# In[52]:


categorical = full.select_dtypes(include = ['object'])


# In[53]:


for column in categorical.columns:
    mode_value = categorical[column].mode()[0]  
    full[column].fillna(mode_value, inplace=True)


# In[54]:


full.isnull().sum().head(40)


# In[55]:


numerical = full.select_dtypes(include = ['int64', 'float64'])
prices = numerical['SalePrice']
numerical = numerical.drop(columns = ['SalePrice'], axis = 1)


# In[56]:


for column in numerical.columns:
    mean_value = numerical[column].mode()[0]  
    full[column].fillna(mean_value, inplace=True)


# In[57]:


full.isnull().sum().head(40)


# In[58]:


full.info()


# # Linear Regression

# In[59]:


from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
import sklearn.inspection
from treeinterpreter import treeinterpreter as ti
from waterfall_chart import plot as waterfall
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.linear_model import LassoLarsIC, Lasso, LassoCV, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error as mse
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF 
from statsmodels.stats.anova import anova_lm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize , poly)


# In[60]:


df = pd.get_dummies(full)
df


# In[61]:


train = df[:1460]
X = train.drop(columns = ['SalePrice'], axis = 1)
y = train['SalePrice']
model = sm.OLS(y, X)
results = model.fit()
summary = summarize(results)


# In[62]:


print(summary)


# In[63]:


X.shape


# In[64]:


y.shape


# In[65]:


X.isnull().sum()


# In[66]:


y.isnull().sum()


# In[67]:


linear = LinearRegression().fit(X, y)


# In[68]:


print(linear.score(X, y))


# In[69]:


predictions = linear.predict(X)
predictions


# In[71]:


mse_train = mse(y, predictions) ** 0.5
mse_train


# # Lasso Regression

# In[72]:


df = pd.get_dummies(full)
df


# In[73]:


train = df[:1460]
X = train.drop(columns = ['SalePrice'], axis = 1)
y = train['SalePrice']


# In[74]:


# for higher dimensional data sets, LassoCV is preferable because the max # of iterations is high enough
model = LassoCV(cv = 5, random_state = 0, max_iter = 10000).fit(X, y)


# In[75]:


model.score(X, y)


# In[76]:


print('R squared training set', round(model.score(X, y)*100, 2))


# In[121]:


predicted = model.predict(X)
mse_train = mean_squared_error(y, predicted) ** 0.5
print('RMSE training set', round(mse_train, 2))

# HUGE MSE!! NOT GOOD


# In[78]:


model.alpha_
# penalty is very aggresive


# In[79]:


AICmodel = LassoLarsIC(criterion = 'aic', max_iter = 10000).fit(X, y)


# In[80]:


BICmodel = LassoLarsIC(criterion = 'bic', max_iter = 10000).fit(X, y)


# In[81]:


lasso_lars_aic = make_pipeline(StandardScaler(), AICmodel)


# In[82]:


lasso_lars_bic = make_pipeline(StandardScaler(), BICmodel)


# In[83]:


results = pd.DataFrame(
    {"alphas": lasso_lars_aic[-1].alphas_,
     "AIC criterion": lasso_lars_aic[-1].criterion_,}).set_index("alphas")
alpha_aic = lasso_lars_aic[-1].alpha_


# In[84]:


results["BIC criterion"] = lasso_lars_bic[-1].criterion_
alpha_bic = lasso_lars_bic[-1].alpha_


# In[85]:


results


# In[86]:


ax = results.plot()
ax.vlines(
    alpha_aic,
    results["AIC criterion"].min(),
    results["AIC criterion"].max(),
    label="alpha: AIC estimate",
    linestyles="--",
    color="tab:blue",)
ax.vlines(
    alpha_bic,
    results["BIC criterion"].min(),
    results["BIC criterion"].max(),
    label="alpha: BIC estimate",
    linestyle="--",
    color="tab:orange",)
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel("criterion")
ax.set_xscale("log")
ax.legend()


# Lasso Regression is not a good model for this data

# # Ridge Regression

# In[87]:


df = pd.get_dummies(full)
df


# In[88]:


train = df[:1460]
X = train.drop(columns = ['SalePrice'], axis = 1)
y = train['SalePrice']


# In[89]:


ridg = Ridge(alpha=1.0)


# In[90]:


ridg.fit(X, y)


# In[91]:


preds = ridg.predict(X)


# In[92]:


rmse_train = mse(y, predicted) ** 0.5
rmse_train


# In[93]:


ridg.score(X, y)


# All Linear regression models are NOT good for this data set. Big MSE!

# # ML Models

# In[95]:


# Import regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# Import evaluation metrics and model selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Import preprocessing
from sklearn.preprocessing import StandardScaler


# In[96]:


df = pd.get_dummies(full)
df


# In[97]:


df.isnull().sum()


# ## Random Forest

# In[184]:


train = df[:1460]
test = df[1460:]


# In[185]:


results = pd.DataFrame()


# In[186]:


X = train.drop(columns = ['SalePrice'], axis = 1)
y = train['SalePrice']


# In[187]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)


# In[188]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[189]:


results['ML Model'] = []
results['RMSE'] = []
results['MAE'] = []
results['MSE'] = []
results['R-squared'] = []


# In[190]:


rf = RandomForestRegressor(random_state = 42)
rf.fit(X_train,y_train)


# In[191]:


features = X.columns


# In[192]:


rmse = mean_squared_error(y_test, rf.predict(X_test)) ** 0.5
mse = mean_squared_error(y_test, rf.predict(X_test))
mae = mean_absolute_error(y_test, rf.predict(X_test))
r_squared = r2_score(y_test, rf.predict(X_test))

print(rmse, mse, mae, r_squared)


# In[193]:


results = results.append(
    {'ML Model' : 'Random Forest', 'RMSE' : rmse, 'MAE' : mae, 'MSE' : mse, 'R-Squared' : r_squared},
    ignore_index = True)


# In[194]:


results = results.drop(columns = ['R-squared'], axis = 1)


# ## Gradient Boosting

# In[196]:


gb = GradientBoostingRegressor()
gb.fit(X_train,y_train)


# In[197]:


rmse = mean_squared_error(y_test, gb.predict(X_test)) ** 0.5
mse = mean_squared_error(y_test, gb.predict(X_test))
mae = mean_absolute_error(y_test, gb.predict(X_test))
r_squared = r2_score(y_test, gb.predict(X_test))

print(rmse, mse, mae, r_squared)


# In[198]:


results = results.append(
    {'ML Model' : 'Gradient Boosting', 'RMSE' : rmse, 'MAE' : mae, 'MSE' : mse, 'R-Squared' : r_squared},
    ignore_index = True)


# ## K-Nearest Regressor

# In[200]:


kn = KNeighborsRegressor()
kn.fit(X_train, y_train)


# In[201]:


rmse = mean_squared_error(y_test, kn.predict(X_test)) ** 0.5
mse = mean_squared_error(y_test, kn.predict(X_test))
mae = mean_absolute_error(y_test, kn.predict(X_test))
r_squared = r2_score(y_test, kn.predict(X_test))

print(rmse, mse, mae, r_squared)


# In[202]:


results = results.append(
    {'ML Model' : 'K-Nearest Regressor', 'RMSE' : rmse, 'MAE' : mae, 'MSE' : mse, 'R-Squared' : r_squared},
    ignore_index = True)


# ## Decision Tree

# In[204]:


dt = DecisionTreeRegressor(random_state = 42)
dt.fit(X_train, y_train)


# In[205]:


rmse = mean_squared_error(y_test, dt.predict(X_test)) ** 0.5
mse = mean_squared_error(y_test, dt.predict(X_test))
mae = mean_absolute_error(y_test, dt.predict(X_test))
r_squared = r2_score(y_test, dt.predict(X_test))

print(rmse, mse, mae, r_squared)


# In[206]:


results = results.append(
    {'ML Model' : 'Decision Tree Regressor', 'RMSE' : rmse, 'MAE' : mae, 'MSE' : mse, 'R-Squared' : r_squared},
    ignore_index = True)


# ## Adaptive Boosting

# In[211]:


ad = AdaBoostRegressor()
ad.fit(X_train, y_train)


# In[212]:


rmse = mean_squared_error(y_test, ad.predict(X_test)) ** 0.5
mse = mean_squared_error(y_test, ad.predict(X_test))
mae = mean_absolute_error(y_test, ad.predict(X_test))
r_squared = r2_score(y_test, ad.predict(X_test))

print(rmse, mse, mae, r_squared)


# In[213]:


results = results.append(
    {'ML Model' : 'Adaptive Boosting', 'RMSE' : rmse, 'MAE' : mae, 'MSE' : mse, 'R-Squared' : r_squared},
    ignore_index = True)


# ## Support Vector Regression

# In[215]:


sv = SVR()
sv.fit(X_train, y_train)


# In[216]:


rmse = mean_squared_error(y_test, sv.predict(X_test)) ** 0.5
mse = mean_squared_error(y_test, sv.predict(X_test))
mae = mean_absolute_error(y_test, sv.predict(X_test))
r_squared = r2_score(y_test, sv.predict(X_test))

print(rmse, mse, mae, r_squared)


# In[217]:


results = results.append(
    {'ML Model' : 'Support Vector Regression', 'RMSE' : rmse, 'MAE' : mae, 'MSE' : mse, 'R-Squared' : r_squared},
    ignore_index = True)


# In[226]:


results = results.sort_values('RMSE', ascending = True)


# In[227]:


results


# ## Making Prediction using Random Forest

# In[228]:


useModel = RandomForestRegressor(random_state = 42)
useModel = useModel.fit(X, y)


# In[229]:


finals = useModel.predict(test.drop(['SalePrice'], axis = 1))


# In[230]:


done = pd.DataFrame()


# In[231]:


done['Id'] = testing['Id']
done['SalePrice'] = finals


# In[232]:


done


# In[ ]:




