#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 17:52:43 2020

@author: amanprasad
"""

import numpy
import pandas
import scipy
import itertools

Data = pandas.read_csv('/Users/amanprasad/Documents/Courses_IIT_Fall_2019/ML/Assignment_ML/HW4/Purchase_Likelihood.csv')
count = Data.groupby('insurance').count()['group_size']
prop = count/ Data.shape[0]
grouped = pandas.DataFrame({'insurance': count.index, 
                                    'Count': count.values, 
                                    'Class Probability': prop.values})
group_cross = pandas.crosstab(Data.insurance, Data.group_size, margins = False, dropna = False)
home_cross = pandas.crosstab(Data.insurance, Data.homeowner, margins = False, dropna = False)
mar_cross = pandas.crosstab(Data.insurance, Data.married_couple, margins = False, dropna = False)

def cramerV(xCat, yCat):
    obsCount = pandas.crosstab(index = xCat, columns = yCat, margins = False, dropna = True)
    cTotal = obsCount.sum(axis = 1)
    rTotal = obsCount.sum(axis = 0)
    nTotal = numpy.sum(rTotal)
    expCount = numpy.outer(cTotal, (rTotal / nTotal))
    
    chiSqStat = ((obsCount - expCount)**2 / expCount).to_numpy().sum()
    
    cramerV = chiSqStat / nTotal
    if (cTotal.size > rTotal.size):
        cramerV = cramerV / (rTotal.size - 1.0)
    else:
        cramerV = cramerV / (cTotal.size - 1.0)
    cramerV = numpy.sqrt(cramerV)

    return(cramerV)
cramer_group= cramerV(Data.group_size, Data.insurance)
cramer_home= cramerV(Data.homeowner, Data.insurance)
cramer_married= cramerV(Data.married_couple, Data.insurance)

def prob(x):
    prob0 = ((grouped['Count'][0] / grouped['Count'].sum()) * 
                   (group_cross[x[0]][0] / group_cross.loc[[0]].sum(axis=1)[0]) * 
                   (home_cross[x[1]][0] / home_cross.loc[[0]].sum(axis=1)[0]) * 
                   (mar_cross[x[2]][0] / mar_cross.loc[[0]].sum(axis=1)[0]))
    prob1 = ((grouped['Count'][1] / grouped['Count'].sum()) * 
                   (group_cross[x[0]][1] / group_cross.loc[[1]].sum(axis=1)[1]) * 
                   (home_cross[x[1]][1] / home_cross.loc[[1]].sum(axis=1)[1]) * 
                   (mar_cross[x[2]][1] / mar_cross.loc[[1]].sum(axis=1)[1]))
    prob2 = ((grouped['Count'][2] / grouped['Count'].sum()) * 
                   (group_cross[x[0]][2] / group_cross.loc[[2]].sum(axis=1)[2]) * 
                   (home_cross[x[1]][2] / home_cross.loc[[2]].sum(axis=1)[2]) * 
                   (mar_cross[x[2]][2] / mar_cross.loc[[2]].sum(axis=1)[2]))
    probs_sum = prob0 + prob1 + prob2
    probfin0 = prob0 / probs_sum
    probfin1 = prob1 / probs_sum
    probfin2 = prob2 / probs_sum

    return [probfin0, probfin1, probfin2]

uni_group = sorted(list(Data.group_size.unique()))
uni_home = sorted(list(Data.homeowner.unique()))
uni_mar = sorted(list(Data.married_couple.unique()))

c = list(itertools.product(uni_group, uni_home, uni_mar))
nbp = []
for i in c:
    t = [prob(i)]
    nbp.extend(t)
    
	
pred=pandas.DataFrame(nbp,columns=['insurance=0','insurance=1','insurance=2'])

Test=pandas.DataFrame();
g=[]
h=[]
m=[]
for i in range(1,5):
	for j in range(2):
		for k in range(2):
			g.append(i)
			h.append(j)
			m.append(k)

Test['group_size']=g
Test['homeowner']=h
Test['married_couple']=m

Test=pandas.concat([Test,pred],axis=1)
print('Predicted Probability:\n',Test )


m=[]
for i in range(len(nbp)):
    temp=nbp[i][1]/nbp[i][0]
    m.append([temp])    
print(numpy.array(m).max())

numpy.array(m).max()
c[numpy.where(m == numpy.array(m).max())[0][0]]