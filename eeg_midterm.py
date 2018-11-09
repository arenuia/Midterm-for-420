 
# coding: utf-8

# # Logistic Regression

import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pylab as plt
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.linear_model import LassoLarsCV
from ast import literal_eval
import operator

import os

cwd = os.getcwd()



#THE THREE CONDITIONS:
#1.The error variable is normally distributed.
#2.The error variance is constant for all values of x.
#3.The errors are independent of each other.
##########################   Part1       #################################################
print("#################   PART1    ###############")
icu = pd.read_csv(os.path.join(cwd, 'icudata.csv'))
#icu.columns = ['AGE', 'RACE', 'CPR', 'SYS', 'HRA' 'TYP']

icu.isnull().sum()

newrace = pd.get_dummies(icu['RACE'], prefix="RACE")
newage = pd.get_dummies(icu['TYP'], prefix="TYP", drop_first=True)
newcpr = pd.get_dummies(icu['CPR'], prefix="CPR", drop_first=True)
icu.drop(['RACE', 'SEX', 'ID', 'TYP', 'CPR'], axis=1, inplace=True)
icu = pd.concat([icu, newrace, newage, newcpr], axis=1)
#print(icu.head())

X = icu.ix[:,(1,2,3,4,5,6,7,8)].values
y = icu.ix[:,0].values

### split data into training, test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=123)
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)

y_pred = LogReg.predict(X_test)
y_prob = LogReg.predict_proba(X_test)

#print(y_prob)
#print(y_pred)

#print(LogReg.intercept_)
#print(1/ (1 + np.exp(LogReg.intercept_)))
#print(LogReg.coef_)
#print(icu.head())




#############   LASSO   ##############
predvar = icu.copy()
target = predvar.STA 
predictors = predvar[['AGE', 'SYS', 'HRA', 'RACE_1', 'RACE_2', 'RACE_3', 'CPR_1', 'TYP_1']].copy()

for i in list(predictors.columns.values):
    predictors[i] = preprocessing.scale(predictors[i].astype('float64'))

pred_train, pred_test, resp_train, resp_test = train_test_split(predictors, target, test_size=.3, random_state=123)
model=LassoLarsCV(cv=10, precompute=True).fit(pred_train,resp_train)
dict(zip(predictors.columns, model.coef_))
m_log_alphascv = -np.log10(model.cv_alphas_)

plt.figure()
plt.plot(m_log_alphascv, model.mse_path_, ':')
plt.plot(m_log_alphascv, model.mse_path_.mean(axis=-1), 'k', label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k', label='alpha CV')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
plt.title('Mean squared error on each fold')
#plt.savefig('Fig02')
#print(pred_train.head())

#plt.show()
print(model.alpha_)
print(model.coef_)
#print(model.intercept_)
#print(pred_train.head())

########################################################################################
###########################   Part 1 RESPONSE            ##############################
#######################################################################################
#(1)
#a.             TABLE:
                #AGE,        SYS,         HRA,          RACE_1,     RACE_2,      RACE_3,      TYP_1,       CPR_1S
                #0.01946696, -0.01645696, -0.00596813, -0.2566194,  -0.23701148, -0.04399663, 1.12856158,  0.87772558

#b.             Viewing the coefficient of CPR, we have 2.41 odds? or about 71% chance of survival TODO: is this right

#c.             Optimal alpha value from the Lasso section: 0.0013716207531124826

#d. 
                #The coefficients are:
                #AGE: 0.05951659
                #SYS: -0.04463656
                #HRA: 0
                #RACE_1: 0
                #RACE_2: 0
                #RACE_3: 0
                #CPR_1: 0.05620125
                #TYP_1: 0.07460764

#########################################################################################
##########################    Part2   #####################################################
print("\n\n#################    PART2     ###############")

teddata = pd.read_csv(os.path.join(cwd, 'ted.csv'), encoding="iso-8859-1")
teddata.drop(['film_date', 'published_date', 'description', 'event', 'related_talks', 'speaker_occupation', 'title', 'url', 'name', 'languages', 'main_speaker'], axis=1, inplace=True)
#leteral_eval turns 'character' "strings" into character objects
teddata['tags'] = teddata['tags'].apply(literal_eval)
teddata['ratings'] = teddata['ratings'].apply(literal_eval)
s = teddata['tags']
s = pd.get_dummies(s.apply(pd.Series).stack(), prefix="TAG_").sum(level=0)
teddata.drop(['tags'], axis=1, inplace=True)
teddata = pd.concat([teddata, s], axis=1)

#2b.
s = teddata['ratings']
s = s.apply(pd.Series)

for i, row in s.iterrows():
    for j in row:
        teddata.loc[i, 'RATING_' + j['name']] = j['count']
teddata.drop(['ratings'], axis=1, inplace=True)
print(teddata.head())

target = teddata.views
predVars = list(teddata.columns.values)
viewsIndex = predVars.index('views')
del predVars[viewsIndex]

predictors = teddata[predVars].copy()
print(predictors.head())

for i in list(predictors.columns.values):
    predictors[i] = preprocessing.scale(predictors[i].astype('float64'))
print(predictors.head())


#2c.

pred_train, pred_test, resp_train, resp_test = train_test_split(predictors, target, test_size=.3, random_state=123)
model = LassoLarsCV(cv=10, precompute=True).fit(pred_train, resp_train)


#2d.
#The optimal value for lambda is 1216.86

print(model.alpha_)


#2e.
#Top 10 betst results produced: magic, body language, success, performance, relationships, live music, time, drones, speech, and youth.

#####   Coefficients:  #######
#('TAG__youth', 11482.475855432787)
#('TAG__speech', 12491.39328459928), 
#('TAG__drones', #15345.191339397952), 
#('TAG__time', 17980.39505624095), 
#('TAG__live music', #25851.14506061892), 
#('TAG__relationships', 31360.70940593261), 
#('TAG__performance', #37747.12675614984), 
#('TAG__success', 53097.30272017584), 
#('TAG__body language', #54897.38713288022), 
#('TAG__magic', 147920.41349920526)]





coefficients = dict(zip(predictors.columns, model.coef_))

tag_prefix = 'TAG_'
tag_coefficients = {k:v for k,v in coefficients.items() if tag_prefix in k}
tag_coefficients = sorted(tag_coefficients.items(), key=operator.itemgetter(1))
print(tag_coefficients)

#2f.
#The worst ratings were: longwinded, confusing, unconvincing, jaw-dropping, obnoxious. Because they all had a coefficient of 0.

rating_prefix = 'RATING_'
rating_coefficients = {k:v for k,v in coefficients.items() if rating_prefix in k}
rating_coefficients = sorted(rating_coefficients.items(), key=operator.itemgetter(1))
print(rating_coefficients)
