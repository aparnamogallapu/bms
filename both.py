# -*- coding: utf-8 -*-
"""
Created on Sat May 11 23:52:13 2019

@author: HP
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir('F://ds//')
dataset=pd.read_csv("C:\\Users\\HP\\Downloads\\internship\\Train_UWu5bXk.csv")
dataset.shape
dataset.columns
list(dataset.columns)
dataset.info()
dataset.describe()
data=dataset.copy()
dataset.isna().sum()
dataset.Item_Identifier.value_counts()
%matplotlib
dataset.isna().sum().plot.bar()

dataset.Item_Weight=dataset.groupby("Item_Identifier")["Item_Weight"].transform(lambda x: x.fillna(x.mean()))
dataset.isna().sum()
dataset[dataset["Item_Weight"].isnull()]
dataset.Item_Weight=dataset["Item_Weight"].transform(lambda x: x.fillna(x.mean()))
dataset.isna().sum()
dataset.Outlet_Size=dataset.groupby("Outlet_Type")["Outlet_Size"].transform(lambda x:x.replace([np.nan],[(x.mode())]))
dataset.isna().sum()
dataset.Outlet_Type.value_counts()
dataset.Item_Visibility=dataset.groupby("Item_Identifier")["Item_Visibility"].transform(lambda x:x.replace(0,x.mean()))
dataset.isna().sum()
dataset.info()
dataset.Item_Fat_Content.unique()
dataset.loc[dataset["Item_Fat_Content"]=='low fat',["Item_Fat_Content"]]='Low Fat'
dataset.loc[dataset["Item_Fat_Content"]=='LF',["Item_Fat_Content"]]='Low Fat'
dataset.loc[dataset["Item_Fat_Content"]=='reg',['Item_Fat_Content']]="Regular"
dataset.Item_Fat_Content.unique()
dataset.columns
dataset.Outlet_Size.unique()
catcols=dataset.select_dtypes(["object"])
for cat in catcols:
    print(cat)
    print(dataset[cat].value_counts())
    print('---'*20)
dataset.info()

'''import os
os.chdir('F:\\ds')
dataset.to_csv('dataetbms.csv',index=False)
dataset1=pd.read_csv('dataetbms.csv')

dataset['ind']=dataset.index'''
########################################### EXPLORATORY DATA ANALYSIS ##################

%matplotlib

################# PAIRPLOT WILL TAKE MORE THAN 1 HOUR ###########################

sns.pairplot(data=dataset[['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility',
       'Item_Type', 'Item_MRP', 'Outlet_Identifier',
       'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type',
       'Outlet_Type','Item_Outlet_Sales']],hue='Item_Outlet_Sales')


############### MULTICOLLINEARITY ##################  
    
colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(dataset.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white',annot=True)

dataset.Item_Identifier.value_counts().sort_values(ascending=False)[0:50].plot.bar()
dataset.Item_Fat_Content.value_counts().sort_values(ascending=False).plot.bar()
dataset.Item_Type.value_counts().sort_values(ascending=False).plot.bar()

dataset.Outlet_Identifier.value_counts().sort_values(ascending=False).plot.barh()
dataset.Outlet_Establishment_Year.value_counts().sort_values(ascending=False).plot.barh()
dataset.Outlet_Size.value_counts().sort_values(ascending=False).plot.barh()

dataset.Outlet_Location_Type.value_counts().sort_values(ascending=False).plot.barh()

dataset.Outlet_Type.value_counts().sort_values(ascending=False).plot.barh()
dataset.groupby(['Item_Type','Item_Fat_Content'])['Item_Type'].count().plot.barh()
dataset.groupby(['Item_Fat_Content','Item_Type'])['Item_Type'].count().plot.barh()
dataset.groupby(['Item_Type','Item_Fat_Content'])['Item_Type'].count().sort_values(ascending=False).plot.barh()


%matplotlib
dataset.groupby(['Outlet_Identifier','Item_Type'])['Item_Type'].count()[0:50].plot.bar()
dataset.groupby(['Item_Type','Outlet_Identifier'])['Item_Type'].count()[0:50].plot.bar()
dataset.groupby(['Item_Type','Outlet_Identifier'])['Item_Type'].count().sort_values(ascending=False).plot.barh()

dataset.groupby(['Outlet_Location_Type','Item_Type'])['Item_Type'].count().plot.bar()
dataset.groupby(['Outlet_Location_Type','Item_Type'])['Item_Type'].count().sort_values(ascending=False).plot.bar()
dataset.groupby(['Outlet_Location_Type','Outlet_Type'])['Item_Type'].count().plot.barh()
dataset.groupby(['Outlet_Location_Type','Outlet_Size'])['Item_Type'].count().plot.bar()
    

dataset.groupby('Item_Fat_Content')['Item_Weight'].min().plot.bar()
dataset.groupby('Item_Fat_Content')['Item_Weight'].max()
dataset.groupby('Item_Fat_Content')['Item_Weight'].mean()
dataset.info()
dataset.Item_Visibility.plot.box()

import numpy as np
np.log(dataset.Item_Visibility).plot.box()

a=dataset.select_dtypes(include=[np.number])
a.info()
a=dataset.select_dtypes(exclude=[np.number])
a.info()

%matplotlib inline
for col in data.select_dtypes(include=[np.number]):
    dataset[col].plot.line()
    plt.title(str(col))
    plt.show()
    
for col in data.select_dtypes(include=[np.number]):
    dataset[col].plot.box()
    plt.title(str(col))
    plt.show()
    
dataset.columns

b=dataset.Item_Fat_Content.unique().tolist()
aa=dataset[dataset.Item_Fat_Content=='Low Fat']['Item_Outlet_Sales'].plot.line()


for cat in dataset.Item_Fat_Content.unique().tolist():    
    dataset[dataset.Item_Fat_Content==str(cat)]['Item_Outlet_Sales'].plot.box()
    plt.title(str(cat))
    plt.show()


%matplotlib
sns.boxplot(data=dataset,y='Item_Fat_Content',x='Item_Outlet_Sales')
sns.boxplot(data=dataset,x='Outlet_Establishment_Year',y='Item_Outlet_Sales')

dataset.info()
dataset.Outlet_Establishment_Year=dataset.Outlet_Establishment_Year.astype('str')

sns.boxplot(data=dataset,x='Outlet_Size',y='Item_Outlet_Sales')
sns.boxplot(data=dataset,x='Outlet_Location_Type',y='Item_Outlet_Sales')
sns.boxplot(data=dataset,y='Outlet_Type',x='Item_Outlet_Sales')
#sns.(data=dataset,x='Item_MRP',y='Item_Outlet_Sales')
sns.scatterplot(data=dataset,x='Item_MRP',y='Item_Outlet_Sales',hue='Item_Type')
sns.scatterplot(data=dataset,x='Item_MRP',y='Item_Outlet_Sales',hue='Outlet_Location_Type')

%matplotlib
sns.distplot(dataset.Item_Outlet_Sales)
sns.distplot(np.log(dataset.Item_Outlet_Sales))

############ MAKING SERIES OBJECT OF "Item_Identifier" AND "Outlet_Identifier" ###############


dataset_Item_Identifier=dataset.Item_Identifier
dataset_Outlet_Identifier=dataset.Outlet_Identifier
dataset=dataset.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1)
dataset.head(10)

dataset['Item_Outlet_Sales'].skew()
dataset['Item_Outlet_Sales']=np.log(dataset['Item_Outlet_Sales'])  
data['Item_Outlet_Sales'].hist()

##############    CREATING DUMMY VARIABLES #######################

dataset_dum=pd.get_dummies(dataset)
catcols = dataset.select_dtypes(['object'])
for cat in catcols:
    print(cat)
    print(dataset[cat].value_counts())
    print('--'*20)

dataset_dum.columns

############### SEPARATING DATA AS DEPENDENT(Y) AND INDEPENDENT(X) #################

    
y=dataset_dum.Item_Outlet_Sales.values
dataset_dum.drop(['Item_Outlet_Sales'],axis=1,inplace=True)
x=dataset_dum.values


############## SCALING THE DATA ##################### OBSERVE DATAFRAME BEFOR N AFTER SCALING ########## 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(x)
x=scaler.transform(x)

############# CHECKING WHETHER DATA SCALED OR NOT ############
x[:6,:]
y[:]

################ SPLITTING THE DATA AS TRAIN AND TEST #################

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size =0.3,random_state=0)

################### APPLYING ALGORITHAM ####################
from sklearn.linear_model import LinearRegression
########## CREATED OBJECT AS MODEL AND USING LINEAR REGRESSION CLASS ##########
model=LinearRegression()
model.fit(x,y)
model.score(x,y)
model.score(x_train,y_train)
model.score(x_test,y_test)
################ APPLYING KFOLD I.E CROSS VALIDATION ##############3

from sklearn.model_selection import cross_val_score,KFold
kfold=KFold(n_splits=10)
score=cross_val_score(model,x,y,cv=kfold)
score
score.mean()

y_pred = model.predict(x_test)
y_pred

########## COMPARE ACTUAL VS PREDICTED
actual=np.exp(y_test)
pridect=np.exp(y_pred)
plt.scatter(model.predict(x),y)
plt.scatter(y_test,y_pred)

############# GENERATING THE RMSS
 
from sklearn.metrics import mean_squared_error
from math import sqrt
rms=sqrt(mean_squared_error(y_test,y_pred))
rms
error=np.exp(rms)
error

############### GENERATING STATISTICAL RESULTS FOR LINEAR REGRESSION
import statsmodels.formula.api as sm
regressor_OLS = sm.OLS(endog=y, exog=x).fit()
regressor_OLS.summary()

'''
#pip install xgboost
import xgboost as xgb
from xgboost import XGBRegressor

xgb=XGBRegressor()
xgb = XGBRegressor(colsample_bytree=0.4,
                 gamma=5.0,                 
                 learning_rate=0.1,
                 max_depth=30,
                 min_child_weight=1,
                 n_estimators=100,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42) 
                                                                                  
                  

xgb.fit(x_train,y_train)
xgb.score(x,y)
xgb.score(x_train,y_train)
xgb.score(x_test,y_test)


from sklearn.model_selection import cross_val_score,KFold
kfold = KFold(n_splits=19,random_state=0)
score = cross_val_score(xgb,x,y,cv=kfold, n_jobs=1)
score
score.mean()

#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV

param_test1 ={
        'n_estimators':[50,100],
        'min_child_weight': [9,10,11],
        'gamma': [0,0.1,1],
        'subsample': [0.6, 0.8, 1.0],
        'max_depth': [1,5,10,15]
        }

model = GridSearchCV(xgb, param_grid=param_test1, n_jobs=1)
model.fit(x_train,y_train)
print("Best Hyper Parameters:",model.best_params_)

model.score(x,y)
model.score(x_train,y_train)
model.score(x_test,y_test)


from sklearn.model_selection import cross_val_score,KFold

kfold = KFold(n_splits=10,random_state=0)
score = cross_val_score(model,x,y,cv=kfold, n_jobs=1)
score.mean()

from sklearn.model_selection import GridSearchCV
params = {'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000], 
          'kernel': ['linear'],'gamma':[0,0.1,0.01,0.001,0.0001]}


model1 = GridSearchCV(xgb, param_grid=params, n_jobs=1)

model1.fit(x_train,y_train)
model1.score(x,y)
model1.score(x_train,y_train)
model1.score(x_test,y_test)
model1.best_params_


xgb.get_params().keys()
from sklearn.model_selection import cross_val_score,KFold
kfold=KFold(n_splits=10)
score=cross_val_score(model,x,y,cv=kfold,n_jobs=-1)
score.mean()'''




############################################# WORKING ON TEST DATA ##########################################
data_test=pd.read_csv("C:\\Users\\HP\\Downloads\\internship\\Test_u94Q5KV.csv")
data_test.shape
data_test.columns
data=data_test.copy()
data_test.info()
data_test.describe()
list(data_test.columns)
data_test.isna().sum()
data_test.Item_Identifier.value_counts()
%matplotlib
data_test.isna().sum().plot.bar()

data_test.Item_Weight= data_test.groupby("Item_Identifier")["Item_Weight"].transform(lambda x: x.fillna(x.mean()))
data_test.isna().sum()
data_test["Item_Weight"].isnull()
data_test[data_test["Item_Weight"].isnull()]
data_test.Item_Weight=data_test["Item_Weight"].transform(lambda x:x.fillna(x.mean()))
data_test.Outlet_Size=data_test.groupby("Outlet_Type")["Outlet_Size"].transform(lambda x:x.replace([np.nan],[x.mode()]))
data_test.Item_Visibility=data_test.groupby("Item_Identifier")["Item_Visibility"].transform(lambda x:x.replace(0,x.mean()))
data_test.isna().sum()
data_test.Item_Fat_Content.unique()
data_test.loc[data_test["Item_Fat_Content"]=='low fat',["Item_Fat_Content"]]='Low Fat'
data_test.loc[data_test["Item_Fat_Content"]=='LF',["Item_Fat_Content"]]='Low Fat'
data_test.loc[data_test["Item_Fat_Content"]=='reg',["Item_Fat_Content"]]='Regular'
data_test.columns
data_test.Outlet_Size.unique()

data_test_Item_Identifier=data_test.Item_Identifier
data_test_Outlet_Identifier=data_test.Outlet_Identifier
data_test=data_test.drop(["Item_Identifier","Outlet_Identifier"],axis=1)
data_test.head(10)

data_test.Outlet_Establishment_Year=data_test.Outlet_Establishment_Year.astype('str')

##############    CREATING DUMMY VARIABLES #######################
data_test_dum=pd.get_dummies(data_test)
data_test_dum.columns
list(data_test_dum.columns)

X_test=data_test_dum.values
X_test=scaler.transform(X_test)

list((data_test_dum.columns==dataset_dum.columns))
(data_test_dum.columns==dataset_dum.columns)

pd.value_counts(list((data_test_dum.columns==dataset_dum.columns)))
model.predict(X_test)

submission_sales=np.exp(model.predict(X_test))
submission_sales_predicted=pd.DataFrame(data = {'Item_Identifier':data_test_Item_Identifier.values, 'Outlet_Identifier':data_test_Outlet_Identifier.values,'Item_Outlet_Sales':submission_sales})

import os
os.chdir('F:\\ds')
submission_sales_predicted.to_csv('SampleSubmissionsales.csv',index = False)
submission_sales_predicted.columns
(submission_sales_predicted.Outlet_Identifier==submission_sales_predicted.Outlet_Identifier).value_counts()




dataset['ind']=dataset.index

