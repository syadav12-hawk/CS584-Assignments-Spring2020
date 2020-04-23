#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name : Sourav Yadav
AID : A20450418
Homework 4
Spring 2020

"""

import numpy
import pandas

import sklearn.naive_bayes as naive_bayes

def RowWithColumn (
    rowVar,          # Row variable
    columnVar,       # Column predictor
    show = 'ROW'):   # Show ROW fraction, COLUMN fraction, or BOTH table

    countTable = pandas.crosstab(index = rowVar, columns = columnVar, margins = False, dropna = True)
 
    return countTable

inData = pandas.read_csv('Purchase_Likelihood.csv', delimiter=',')
subData = inData[['group_size', 'homeowner', 'married_couple', 'insurance']].dropna()


count = inData['insurance'].value_counts()
total = inData['insurance'].count()
frequencyCount = count
print(frequencyCount)
probability = count/total
print(probability)

cat_group_size = subData['group_size'].unique()
cat_homeowner = subData['homeowner'].unique()
cat_married_couple = subData['married_couple'].unique()
cat_insurance = subData['insurance'].unique()

print('Unique Values of group_size: \n', cat_group_size)
print('Unique Values of homeowner: \n', cat_homeowner)
print('Unique Values of married_couple: \n', cat_married_couple)
print('Unique Values of insurance: \n', cat_insurance)

gstable = RowWithColumn(rowVar = subData['group_size'], columnVar = subData['insurance'], show = 'ROW')
hotable = RowWithColumn(rowVar = subData['homeowner'], columnVar = subData['insurance'], show = 'ROW')
mctable = RowWithColumn(rowVar = subData['married_couple'], columnVar = subData['insurance'], show = 'ROW')

print(gstable)
print(hotable)
print(mctable)



def cramerV (
    rowVar,          # Row variable
    columnVar,       # Column predictor
    show = 'ROW'):   # Show ROW fraction, COLUMN fraction, or BOTH table

    countTable = pandas.crosstab(index = rowVar, columns = columnVar, margins = False, dropna = True)
    cTotal = countTable.sum(axis = 1)
    rTotal = countTable.sum(axis = 0)
    nTotal = numpy.sum(rTotal)
    expCount = numpy.outer(cTotal, (rTotal / nTotal))

    chiSqStat = ((countTable - expCount)**2 / expCount).to_numpy().sum()

    cramerVStat = chiSqStat / nTotal
    if (cTotal.size > rTotal.size):
        cramerVStat = cramerVStat / (rTotal.size - 1.0)
    else:
        cramerVStat = cramerVStat / (cTotal.size - 1.0)
    cramerVStat = numpy.sqrt(cramerVStat)
    print(cramerVStat)

print('CramerV Statistics')
print('group_size :')
cramerV(rowVar = subData['group_size'], columnVar = subData['insurance'], show = 'ROW')
print('homeowner : ')
cramerV(rowVar = subData['homeowner'], columnVar = subData['insurance'], show = 'ROW')
print('married_couple : ')
cramerV(rowVar = subData['married_couple'], columnVar = subData['insurance'], show = 'ROW')

subData = subData.astype('category')
xTrain = subData[['group_size', 'homeowner', 'married_couple']]

yTrain = subData['insurance']

_objNB = naive_bayes.MultinomialNB(alpha = 1e-10)
thisModel = _objNB.fit(xTrain, yTrain)

print('Empirical probability of features given a class, P(x_i|y)')
print(xTrain.columns)
print(numpy.exp(_objNB.feature_log_prob_))
print('\n')

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

predpob=pandas.DataFrame(_objNB.predict_proba(df),columns=['0','1','2'])
        
predpob['div']=predpob['1']/predpob['0']
print('Max Odd Value:',max(predpob['div']))
print('Value of Combination',myList[predpob['div'].idxmax(axis=0)])


