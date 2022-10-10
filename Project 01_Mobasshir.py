#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV as RandomCV


# In[3]:


df=pd.read_csv(r'train.csv')


# In[4]:


df.head()


# In[5]:


df.info() 


# In[6]:


# No missing values for any variables


# In[6]:


df.describe()


# ## Density plot for dependent variable

# In[7]:


df['critical_temp'].describe()


# In[8]:


fig, ax = plt.subplots(figsize=(10, 7))
n, bins, patches = plt.hist(x=df['critical_temp'], density=True, bins='auto', color='#0300bb',
                            alpha=0.7, rwidth=0.85)

plt.xlabel('Critical Temperature (K)',size=18)
plt.ylabel('Density',size=18)
plt.axis("tight")


# ## Correlation plots

# In[9]:


plt.figure(figsize=(10,8))

mask = np.triu(np.ones_like(df.corr(), dtype=bool))
sns.heatmap(df.corr(), mask=mask, vmin=-1, vmax=1, cmap='viridis')


# In[28]:


# Predictors are highly correlated, or multicollinearity exists among the predictors


# In[10]:


corr_matrix = df.corr()
corr_matrix["critical_temp"].sort_values(ascending=False)


# ## Feature selection by decision tree model (XGBoost)

# In[11]:


x_vals=df.copy()

x_vals.drop('critical_temp',1,inplace=True)

y_vals=df['critical_temp']


# In[12]:


#spliting data_set into test and train set 
X_train, X_test, Y_train, Y_test = train_test_split(x_vals, y_vals, test_size=0.3, random_state=0)


# In[13]:


from xgboost import XGBRegressor

model = XGBRegressor(random_state=0)
model.fit(X_train, Y_train)
importances = pd.DataFrame(data={
    'Attribute': X_train.columns,
    'Importance': model.feature_importances_
})
importances = importances.sort_values(by='Importance', ascending=False)


# In[14]:


importances


# In[15]:


plt.bar(x=importances['Attribute'][0:20], height=importances['Importance'][0:20], color='g')
plt.title('Feature importances obtained from XGBoost', size=15)
plt.xticks(rotation='vertical')
plt.ylabel("Feature importance",size=12)
plt.show()


# In[16]:


from xgboost import plot_importance
plot_importance(model)


# In[17]:


max(model.feature_importances_)


# In[18]:


min(model.feature_importances_)


# ### Pairplots based on top 10 features

# In[19]:


importances_10 = importances.head(10)
feature_list=list(importances_10['Attribute'])
df_pplot=df.filter(feature_list)


# In[20]:


feature_list


# In[21]:


sns.pairplot(df_pplot.loc[:,df_pplot.dtypes == 'float64'])
plt.show()


# ### Pearson Corelation Matrix for the first 10 most important features

# In[22]:


fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(df_pplot.corr(),annot=True, fmt=".2f")
plt.axis('tight')
plt.savefig('corelation_10.png')

plt.show()


# ## Filtering only the important features: threshold 0.001

# In[23]:


importances_new=importances[importances['Importance']>0.001]

to_filter=list(importances_new['Attribute'])
x_vals_new=x_vals.filter(to_filter)


# In[24]:


len(importances_new)


# In[25]:


importances_new


# ### filtering all Highly correlated variables

# In[26]:


# Create correlation matrix
corr_matrix_new = x_vals_new.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix_new.where(np.triu(np.ones(corr_matrix_new.shape), k=1).astype(bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] >= 0.95)]


# In[27]:


to_drop


# In[28]:


x_vals_new1 = x_vals_new.drop(['wtd_std_ThermalConductivity', 'wtd_mean_Valence','std_atomic_radius', 'range_Density', 'std_Valence', 'wtd_gmean_Density'], axis=1)


# In[30]:


x_vals_new1


# ### Linear Regression Model 

# In[29]:


# with all 81 variables
from sklearn.linear_model import LinearRegression
regr = LinearRegression()
  
regr.fit(X_train, Y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

print(regr.score(X_test, Y_test))


# In[30]:


from sklearn.metrics import mean_squared_error

rmse = mean_squared_error(Y_test, y_pred, squared=False)

print(rmse)


# In[31]:


fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(Y_test,y_pred,'.')
x=[0, 140]
y = [0, 140]
ax.plot(x,y,'r')
plt.title('Multiple linear regression model', size=15)
plt.xlabel("Observed Critical Temperature (K)",size=12)
plt.ylabel("Predicted Critical Temperature (K)",size=12)
plt.axis('tight')


# In[32]:


# with 37 variables
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(x_vals_new, y_vals, test_size=0.3, random_state=0)


# In[33]:


from sklearn.linear_model import LinearRegression
regr = LinearRegression()
  
regr.fit(X_train1, Y_train1)

# Make predictions using the testing set
y_pred1 = regr.predict(X_test1)

print(regr.score(X_test1, Y_test1))


# In[34]:


from sklearn.metrics import mean_squared_error

rmse1 = mean_squared_error(Y_test1, y_pred1, squared=False)

print(rmse1)


# In[35]:


fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(Y_test1,y_pred1,'.')
x=[0, 140]
y = [0, 140]
ax.plot(x,y,'r')
plt.title('Multiple linear regression model', size=15)
plt.xlabel("Observed Critical Temperature (K)",size=12)
plt.ylabel("Predicted Critical Temperature (K)",size=12)
plt.axis('tight')


# In[ ]:





# In[37]:


# with 31 variables
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(x_vals_new1, y_vals, test_size=0.3, random_state=0)


# In[38]:


from sklearn.linear_model import LinearRegression
regr = LinearRegression()
  
regr.fit(X_train2, Y_train2)

# Make predictions using the testing set
y_pred2 = regr.predict(X_test2)

#print(regr.score(X_test2, Y_test2))
print(r2_score(Y_test2, y_pred2))


# In[39]:


rmse_linreg = mean_squared_error(Y_test2, y_pred2, squared=False)

print(rmse_linreg)


# In[40]:


fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(Y_test2,y_pred2,'.')
x=[0, 140]
y = [0, 140]
ax.plot(x,y,'r')
plt.title('Multiple linear regression model', size=15)
plt.xlabel("Observed Critical Temperature (K)",size=12)
plt.ylabel("Predicted Critical Temperature (K)",size=12)
plt.axis('tight')


# ## Statistical way to compute linear regression

# In[41]:


# with all 81 variables
x_vals['Intercept']=1


# In[42]:


# with all 81 variables
import statsmodels.api as sm

model = sm.OLS(y_vals,x_vals)

res = model.fit()

print(res.summary())


# In[43]:


# with 37 variables
x_vals_new['Intercept']=1


# In[44]:


import statsmodels.api as sm

model = sm.OLS(y_vals,x_vals_new)

res = model.fit()

print(res.summary())


# In[45]:


# with 37 variables
x_vals_new1['Intercept']=1


# In[46]:


import statsmodels.api as sm

model = sm.OLS(y_vals,x_vals_new1)

res = model.fit()

print(res.summary())


# In[47]:


def myprint(s):
    with open(r'modelsummary34.txt','w+') as f:
        print(s, file=f)

myprint(res.summary())


# ## Prediction with the Linear regression model

# In[48]:


Y = res.predict(x_vals_new1)

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(y_vals,Y,'.')
x=[0, 140]
y = [0, 120]
ax.plot(x,y,'r')
plt.title('Multiple linear regression model', size=15)
plt.xlabel("Observed Critical Temperature (K)",size=12)
plt.ylabel("Predicted Critical Temperature (K)",size=12)

plt.axis('tight')


# ## Ridge Regression Model (check Big Data codes from FSU)

# In[49]:


from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, make_scorer


# In[50]:


from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
ridge.fit(X_train2, Y_train2)
alpha = ridge.alpha_
#print("Best alpha :", alpha)

#print("Try again for more precision with alphas centered around " + str(alpha))
ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 
                cv = 10)
ridge.fit(X_train2, Y_train2)
alpha = ridge.alpha_
#print("Best alpha :", alpha)

#print("Ridge RMSE on Training set :", rmse_cv_train(ridge).mean())
#print("Ridge RMSE:", rmse_cv_test(ridge).mean())
#print("Intercept: ", ridge.intercept_)
#print("Coefficients: ", ridge.coef_)
#y_train_rdg = ridge.predict(X_train)
y_test_rdg = ridge.predict(X_test2)

print("ridge_r2 =", r2_score(Y_test2, y_test_rdg))

rmse_ridge = mean_squared_error(Y_test2, y_test_rdg, squared=False)

print('ridge_rmse = ', rmse_ridge)


# ## Lasso Regression

# In[51]:


lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 
                          0.3, 0.6, 1], 
                max_iter = 50000, cv = 10)
lasso.fit(X_train2, Y_train2)
alpha = lasso.alpha_
#print("Best alpha :", alpha)

#print("Try again for more precision with alphas centered around " + str(alpha))
lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, 
                          alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, 
                          alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, 
                          alpha * 1.4], 
                max_iter = 50000, cv = 10)

lasso.fit(X_train2, Y_train2)
alpha = lasso.alpha_
#print("Best alpha :", alpha)

#print("Lasso RMSE on Training set :", rmse_cv_train(lasso).mean())
#print("Lasso RMSE on Test set :", rmse_cv_test(lasso).mean())
#print("Intercept: ", lasso.intercept_)
#print("Coefficients: ", lasso.coef_)
#y_train_las = lasso.predict(X_train)
y_test_las = lasso.predict(X_test2)

print("Lasso_r2 =", r2_score(Y_test2, y_test_las))

rmse_lasso = mean_squared_error(Y_test2, y_test_las, squared=False)

print('ridge_rmse = ', rmse_lasso)


# ## Random Forest Regression

# In[57]:


rnf = RandomForestRegressor(random_state=0)
rnf.fit(X_train2, Y_train2)
rnf_train=rnf.score(X_train2, Y_train2)
rnf_test=rnf.score(X_test2, Y_test2)
#print("Accuracy on training set: {:.3f}".format(rnf.score(X_train2, Y_train2)))
#print("Accuracy on test set: {:.3f}".format(rnf.score(X_test2, Y_test2)))
Y_pred_ran=rnf.predict(X_test2)
rmse_rnf= sqrt(mean_squared_error(Y_test2, Y_pred_ran))
print("r2=", r2_score(Y_test2, Y_pred_ran))

print("RMSE=",rmse_rnf)


# ## Gradient Boosting Regression

# In[58]:


gbr = GradientBoostingRegressor(random_state=0)
gbr=gbr.fit(X_train2, Y_train2)
gbr_train=gbr.score(X_train2, Y_train2)
gbr_test=gbr.score(X_test2, Y_test2)
#print("Accuracy on training set: {:.3f}".format(gbr.score(X_train2, Y_train2)))
#print("Accuracy on test set: {:.3f}".format(gbr.score(X_test2, Y_test2)))
Y_pred=gbr.predict(X_test2)
rmse_gbr= sqrt(mean_squared_error(Y_test2, Y_pred))
print("r2=", r2_score(Y_test2, Y_pred))
print("RMSE=",rmse_gbr)


# ## Knearest Neigbour Regressor

# In[59]:


from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=5,weights='distance') 
model=knn.fit(X_train2, Y_train2)
Y_pred=model.predict(X_test2)
rmse_Knn= sqrt(mean_squared_error(Y_test2, Y_pred))
print("r2=", r2_score(Y_test2, Y_pred))
print("RMSE=",rmse_Knn)


# ## XGBoost Regression

# In[60]:


import xgboost as xg
xgb_r= XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
model= xgb_r.fit(X_train2, Y_train2)
Y_pred=model.predict(X_test2)
rmse_XGBoost= sqrt(mean_squared_error(Y_test2, Y_pred))
print("r2=", r2_score(Y_test2, Y_pred))
print("RMSE=",rmse_XGBoost)


# ## Winner is Random Forest Regression (High R_squared value and low RMSE)

# In[56]:


fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(Y_test2,Y_pred_ran,'.')
x=[0, 140]
y = [0, 140]
ax.plot(x,y,'r')
plt.title('Random Forest regression model', size=15)
plt.xlabel("Observed Critical Temperature (K)",size=12)
plt.ylabel("Predicted Critical Temperature (K)",size=12)
plt.axis('tight')


# ### Model Comparison

# In[78]:


RMSE=[rmse_linreg,rmse_ridge,rmse_lasso, rmse_rnf,rmse_gbr,rmse_Knn,rmse_XGBoost]
RMSE


# In[76]:


algorithm_type = [1,2,3,4,5,6,7]
LABELS = ["Linear Regression", "Ridge", "Lasso","Random Forest","Gradient Boosting","KNN","XGBoost"]
fig, ax = plt.subplots(figsize=(8, 6))
plt.bar(algorithm_type, RMSE, width=0.5,align='center',color='c')
plt.xticks(algorithm_type, LABELS,size=12,rotation='vertical')
plt.title("Model Comparison", size=18)
plt.xlabel("Types of Regressors",size=15)
plt.ylabel("Temperature RMSE (K)",size=15)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Feature selection by Gradient Boosting Regressor

# In[25]:


model = GradientBoostingRegressor(random_state=0)
model.fit(X_train, Y_train)
importances = pd.DataFrame(data={
    'Attribute': X_train.columns,
    'Importance': model.feature_importances_
})
importances = importances.sort_values(by='Importance', ascending=False)


# In[26]:


plt.bar(x=importances['Attribute'][0:20], height=importances['Importance'][0:20], color='#087E8B')
plt.title('Feature importances obtained from coefficients', size=20)
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:





# In[ ]:




