#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 13:56:12 2020

@author: gauravsinghania
"""

import pandas
import statsmodels.api as stats
import numpy

import scipy
import sympy 
import math

myData = pandas.read_csv('Purchase_Likelihood.csv',delimiter=',')
myData = myData.dropna()

# A function that find the non-aliased columns, fit a logistic model, and return the full parameter estimates
def build_mnlogit (fullX, y):
    # Number of all parameters
    nFullParam = fullX.shape[1]

    # Number of target categories
    y_category = y.cat.categories
    nYCat = len(y_category)

    # Find the non-redundant columns in the design matrix fullX
    reduced_form, inds = sympy.Matrix(fullX.values).rref()

    # These are the column numbers of the non-redundant columns
    #if (debug == 'Y'):
    print('Column Numbers of the Non-redundant Columns:')
    print(inds)

    # Extract only the non-redundant columns for modeling
    X = fullX.iloc[:, list(inds)]

    # The number of free parameters
    thisDF = len(inds) * (nYCat - 1)

    # Build a multionomial logistic model
    logit = stats.MNLogit(y, X)
    thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
    thisParameter = thisFit.params
    thisLLK = logit.loglike(thisParameter.values)

    
    print(thisFit.summary())
    print("Model Parameter Estimates:\n", thisParameter)
    print("Model Log-Likelihood Value =", thisLLK)
    print("Number of Free Parameters =", thisDF)

    # Recreat the estimates of the full parameters
    workParams = pandas.DataFrame(numpy.zeros(shape = (nFullParam, (nYCat - 1))))
    workParams = workParams.set_index(keys = fullX.columns)
    fullParams = pandas.merge(workParams, thisParameter, how = "left", left_index = True, right_index = True)
    fullParams = fullParams.drop(columns = '0_x').fillna(0.0)

    # Return model statistics
    return (thisLLK, thisDF, fullParams)

y = myData['insurance'].astype('category')

xG = pandas.get_dummies(myData[['group_size']].astype('category'))
xH = pandas.get_dummies(myData[['homeowner']].astype('category'))
xM = pandas.get_dummies(myData[['married_couple']].astype('category'))

# Intercept only model
designX = pandas.DataFrame(y.where(y.isnull(), 1))
LLK0, DF0, fullParams0 = build_mnlogit (designX, y,)

# Intercept + group_size
designX = stats.add_constant(xG, prepend=True)
LLK_1G, DF_1G, fullParams_1G = build_mnlogit (designX, y)
testDev = 2 * (LLK_1G - LLK0)
testDF = DF_1G - DF0
testPValue = scipy.stats.chi2.sf(testDev, testDF)
featureImportance = math.log10(testPValue)* -1

print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
print(' Feature Importance = ', featureImportance)

# Intercept + group_size + homeowner
designX = xG
designX = designX.join(xH)
designX = stats.add_constant(designX, prepend=True)
LLK_1G_1H, DF_1G_1H, fullParams_1G_1H = build_mnlogit (designX, y)
testDev = 2 * (LLK_1G_1H - LLK_1G)
testDF = DF_1G_1H - DF_1G
testPValue = scipy.stats.chi2.sf(testDev, testDF)

print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)


# Intercept + group_size + homeowner + married_couple
designX = xG
designX = designX.join(xH)
designX = designX.join(xM)
designX = stats.add_constant(designX, prepend=True)
LLK_1G_1H_1M, DF_1G_1H_1M, fullParams_1G_1H_1M = build_mnlogit (designX, y)
testDev = 2 * (LLK_1G_1H_1M - LLK_1G_1H)
testDF = DF_1G_1H_1M - DF_1G_1H
testPValue = scipy.stats.chi2.sf(testDev, testDF)
featureImportance = math.log10(testPValue)* -1
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
print(' Feature Importance = ', featureImportance)

# A function that returns the columnwise product of two dataframes (must have same number of rows)
def create_interaction (inDF1, inDF2):
    name1 = inDF1.columns
    name2 = inDF2.columns
    outDF = pandas.DataFrame()
    for col1 in name1:
        for col2 in name2:
            outName = col1 + " * " + col2
            outDF[outName] = inDF1[col1] * inDF2[col2]
    return(outDF)


# Intercept + group_size + homeowner + married_couple + group_size * homeowner
designX = xG
designX = designX.join(xH)
designX = designX.join(xM)

# Create the columns for the group_size * homeowner interaction effect
xGH = create_interaction(xG, xH)
designX = designX.join(xGH)

designX = stats.add_constant(designX, prepend=True)
LLK_2GH, DF_2GH, fullParams_2GH = build_mnlogit (designX, y)
testDev = 2 * (LLK_2GH - LLK_1G_1H_1M)
testDF = DF_2GH - DF_1G_1H_1M
testPValue = scipy.stats.chi2.sf(testDev, testDF)
featureImportance = math.log10(testPValue)* -1
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
print(' Feature Importance = ', featureImportance)

# Intercept + group_size + homeowner + married_couple + group_size * homeowner + group_size * married_couple
designX = xG
designX = designX.join(xH)
designX = designX.join(xM)

# Create the columns for the group_size * homeowner interaction effect
xGH = create_interaction(xG, xH)
designX = designX.join(xGH)

# Create the columns for the homeowner * married_couple interaction effect
xGM = create_interaction(xG, xM)
designX = designX.join(xGM)

designX = stats.add_constant(designX, prepend=True)
LLK_2G_1H_1M, DF_2G_1H_1M, fullParams_2GHM = build_mnlogit (designX, y)
testDev = 2 * (LLK_2G_1H_1M - LLK_2GH)
testDF = DF_2G_1H_1M - DF_2GH
testPValue = scipy.stats.chi2.sf(testDev, testDF)
featureImportance = math.log10(testPValue)* -1
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
print(' Feature Importance = ', featureImportance)


# Intercept + group_size + homeowner + married_couple + group_size * homeowner + group_size * married_couple + homeowner * married_couple
designX = xG
designX = designX.join(xH)
designX = designX.join(xM)

# Create the columns for the group_size * homeowner interaction effect
xGH = create_interaction(xG, xH)
designX = designX.join(xGH)

# Create the columns for the homeowner * married_couple interaction effect
xGM = create_interaction(xG, xM)
designX = designX.join(xGM)

xHM = create_interaction(xH,xM)
designX = designX.join(xHM)

designX = stats.add_constant(designX, prepend=True)
LLK_2GHM, DF_2GHM, fullParams_2GHM = build_mnlogit (designX, y,)
testDev = 2 * (LLK_2GHM - LLK_2G_1H_1M)
testDF = DF_2GHM - DF_2G_1H_1M
testPValue = scipy.stats.chi2.sf(testDev, testDF)
featureImportance = math.log10(testPValue)* -1
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
print(' Feature Importance = ', featureImportance)





#Question 2

myList = [[1,0,0],
          [1,0,1],
          [1,1,0],
          [1,1,1],
          [2,0,0],
          [2,0,1],
          [2,1,0],
          [2,1,1],
          [3,0,0],
          [3,0,1],
          [3,1,0],
          [3,1,1],
          [4,0,0],
          [4,0,1],
          [4,1,0],
          [4,1,1]]

df = pandas.DataFrame(myList, columns=['group_size', 'homeowner', 'married_couple'])

g = pandas.get_dummies(df[['group_size']].astype('category'))
h = pandas.get_dummies(df[['homeowner']].astype('category'))
m = pandas.get_dummies(df[['married_couple']].astype('category'))
gh = create_interaction(g,h)
gm = create_interaction(g,m)
hm = create_interaction(h,m)

df1 = g.join(h)
df1 = df1.join(m)
df1 = df1.join(gh)
df1 = df1.join(gm)
df1 = df1.join(hm)
df1 = stats.add_constant(df1, prepend=True)

# Build a multionomial logistic model
logit = stats.MNLogit(y, designX)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params

print(thisParameter)

Probabilities = thisFit.predict(df1)
print(Probabilities)

maximum = 0
for i in range(len(Probabilities)):
    oddValue = Probabilities[1][i]/Probabilities[0][i]
    if oddValue > maximum:
        maximum = oddValue
        k = i
        
print('Max Odd Value:',maximum)
print('Value of Combination',myList[k])



