#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 23:11:59 2020

@author: amanprasad
"""


# #  Assignment 4 : Machine Learning Question 1



#Import required libraries and dataset
import numpy
import pandas
import scipy
import sympy 
import math

import statsmodels.api as stats

dataframe = pandas.read_csv('C:/Users/soura/OneDrive/Desktop/ML/ML Spring20_Assignemnt/HW4/Purchase_Likelihood.csv')

dataframe = dataframe.dropna()

# Specify Origin as a categorical variable
y = dataframe['insurance'].astype('category')




# Specify Group_size, Homeowner and married_couple as categorical variables
d_group_size = pandas.get_dummies(dataframe[['group_size']].astype('category'))
d_homeowner = pandas.get_dummies(dataframe[['homeowner']].astype('category'))
d_married_couple = pandas.get_dummies(dataframe[['married_couple']].astype('category'))





#Function to build MNL model
def build_mnlogit (fullX, y, debug = 'N'):
    # Number of all parameters
    nFullParam = fullX.shape[1]

    # Number of target categories
    y_category = y.cat.categories
    nYCat = len(y_category)

    # Find the non-redundant columns in the design matrix fullX
    reduced_form, inds = sympy.Matrix(fullX.values).rref()

    # These are the column numbers of the non-redundant columns
    if (debug == 'Y'):
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

    if (debug == 'Y'):
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





# Intercept only model
designX = pandas.DataFrame(y.where(y.isnull(), 1))
LLK0, DF0, fullParams0 = build_mnlogit (designX, y, debug = 'Y')
(LLK0,DF0)
print(LLK0)
print(DF0)
print(fullParams0)




# Intercept + Group_size
designX = stats.add_constant(d_group_size, prepend=True)
LLK_1R, DF_1R, fullParams_1R = build_mnlogit (designX, y, debug = 'Y')
testDev = 2 * (LLK_1R - LLK0)
testDF = DF_1R - DF0
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)




# Intercept + Group_size + homeowner
designX = d_group_size
designX = designX.join(d_homeowner)
designX = stats.add_constant(designX, prepend=True)
LLK_1R_1J, DF_1R_1J, fullParams_1R_1J = build_mnlogit (designX, y, debug = 'Y')
testDev = 2 * (LLK_1R_1J - LLK_1R)
testDF = DF_1R_1J - DF_1R
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)





# Intercept + Group_size + homeowner + married_couple
designX = d_group_size
designX = designX.join(d_homeowner)
designX = designX.join(d_married_couple)
designX = stats.add_constant(designX, prepend=True)

LLK_1R_1J_1M, DF_1R_1J_1M, fullParams_1R_1J_1M = build_mnlogit (designX, y, debug = 'Y')
testDev = 2 * (LLK_1R_1J_1M - LLK_1R_1J)
testDF = DF_1R_1J_1M - DF_1R_1J
testPValue = scipy.stats.chi2.sf(testDev, testDF)

print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)





def create_interaction (inDF1, inDF2):
    name1 = inDF1.columns
    name2 = inDF2.columns
    outDF = pandas.DataFrame()
    for col1 in name1:
        for col2 in name2:
            outName = col1 + " * " + col2
            outDF[outName] = inDF1[col1] * inDF2[col2]
    return(outDF)





# Intercept + Group_size + homeowner + married_couple + Group_size*homeowner
designX = d_group_size
designX = designX.join(d_homeowner)
designX = designX.join(d_married_couple)

d_gs_ho = create_interaction(d_group_size, d_homeowner)
designX = designX.join(d_gs_ho)
designX = stats.add_constant(designX, prepend=True)

LLK_2RJ, DF_2RJ, fullParams_2RJ = build_mnlogit (designX, y, debug = 'Y')
testDev = 2 * (LLK_2RJ - LLK_1R_1J_1M)
testDF = DF_2RJ - DF_1R_1J_1M
testPValue = scipy.stats.chi2.sf(testDev, testDF)

print('Deviance Chi=Square Test') 
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)




# Intercept + Group_size + homeowner + married_couple + Group_size*homeowner + Group_size*married_couple
designX = d_group_size
designX = designX.join(d_homeowner)
designX = designX.join(d_married_couple)

d_gs_ho = create_interaction(d_group_size, d_homeowner)
designX = designX.join(d_gs_ho)
designX = stats.add_constant(designX, prepend=True)

xRJ1 = create_interaction(d_group_size, d_married_couple)
designX = designX.join(xRJ1)
designX = stats.add_constant(designX, prepend=True)

LLK_2RJ, DF_2RJ, fullParams_2RJ = build_mnlogit (designX, y, debug = 'Y')
testDev = 2 * (LLK_2RJ - LLK_1R_1J_1M)
testDF = DF_2RJ - DF_1R_1J_1M
testPValue = scipy.stats.chi2.sf(testDev, testDF)

print('Deviance Chi=Square Test') 
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)




# Intercept + Group_size + homeowner + married_couple + Group_size*homeowner + Group_size*married_couple + homeowner*married_couple
designX = d_group_size
designX = designX.join(d_homeowner)
designX = designX.join(d_married_couple)

d_gs_ho = create_interaction(d_group_size, d_homeowner)
designX = designX.join(d_gs_ho)
designX = stats.add_constant(designX, prepend=True)

xRJ1 = create_interaction(d_group_size, d_married_couple)
designX = designX.join(xRJ1)
designX = stats.add_constant(designX, prepend=True)

d_ho_mc = create_interaction(d_homeowner, d_married_couple)
designX = designX.join(d_ho_mc)
designX = stats.add_constant(designX, prepend=True)

LLK_2JM, DF_2JM, fullParams_2JM = build_mnlogit (designX, y, debug = 'Y')
testDev = 2 * (LLK_2JM - LLK_2RJ)
testDF = DF_2JM - DF_2RJ
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)





#a) (5 points) List the aliased parameters that you found in your model.
'''
lis = [4,6,8,10,12,14,15,16,18,19,20]
#designX.columns[n]
print("Aliased parameters in the model")
for i in lis:
    print(designX.columns[i])
'''

print("List the aliased columns that you found in your model matrix.\n", fullParams_2JM)


# d) (5 points) Calculate the Feature Importance Index as the negative base-10 logarithm of the significance value.  List your indices by the model effects.

feature_Imps = [4.347870389027117e-210,4.306457217534288e-19,5.512105969198056e-52,4.13804354648637e-16 ]
for i in feature_Imps:
    print(-math.log10(i))





#Build the final model
logit = stats.MNLogit(y, designX)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)





#Create all possible combinations for values in each variable
group_sizes = sorted(list(dataframe.group_size.unique()))
homeowners = sorted(list(dataframe.homeowner.unique()))
married_couples = sorted(list(dataframe.married_couple.unique()))

import itertools
all_combi = list(itertools.product(group_sizes, homeowners, married_couples))

df = pandas.DataFrame(all_combi, columns=['group_size','homeowner','married_couple'])

df_groupsize = pandas.get_dummies(df[['group_size']].astype('category'))
X_Test = df_groupsize

df_homeowner = pandas.get_dummies(df[['homeowner']].astype('category'))
X_Test = X_Test.join(df_homeowner)

df_marriedcouple = pandas.get_dummies(df[['married_couple']].astype('category'))
X_Test = X_Test.join(df_marriedcouple)

df_groupsize_h = create_interaction(df_groupsize, df_homeowner)
df_groupsize_h = pandas.get_dummies(df_groupsize_h)
X_Test = X_Test.join(df_groupsize_h)


xRJ1 = create_interaction(d_group_size, d_married_couple)
xRJ1 = pandas.get_dummies(xRJ1)
X_Test = X_Test.join(xRJ1)


df_homeowner_m = create_interaction(df_homeowner, df_marriedcouple)
df_homeowner_m = pandas.get_dummies(df_homeowner_m)
X_Test = X_Test.join(df_homeowner_m)


X_Test = stats.add_constant(X_Test, prepend=True)





# Predicted Probabilities
#(e)
#For each of the sixteen possible value combinations of the three features, calculate the predicted probabilities for A = 0, 1, 2 based on the multinomial logistic model.
#List your answers in a table with proper labelling.

predictions = thisFit.predict(X_Test)
pandas.DataFrame.join(pandas.DataFrame(all_combi, columns = ["group_size","homeOwner","Married_couple"]),predictions)





# f) (5 points) Based on your model, what values of group_size, homeowner, and married_couple will maximize the odds value Prob(A=1) / Prob(A = 0)?  
# What is that maximum odd value?
#Odds value for Prob(A=1) / Prob(A = 0)
(predictions[1]/predictions[0])





print("The observed max value is for combination is",all_combi[3])
print("The maximum odd value is",max((predictions[1]/predictions[0])))





# g) (5 points) Based on your model, what is the odds ratio for group_size = 3 versus group_size = 1, and A = 2 versus A = 0?  
# Mathematically, the odds ratio is (Prob(A=2)/Prob(A=0) | group_size = 3) / ((Prob(A=2)/Prob(A=0) | group_size = 1).





fullParams_2JM

