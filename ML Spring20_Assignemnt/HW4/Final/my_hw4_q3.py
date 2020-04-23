
"""
Name : Sourav Yadav
AID : A20450418
Homework 4
Spring 2020

"""

import numpy
import pandas

import itertools
input_Data = pandas.read_csv('Purchase_Likelihood.csv', delimiter=',')
Data = input_Data[['group_size', 'homeowner', 'married_couple', 'insurance']].dropna()


count = input_Data['insurance'].value_counts()
total = input_Data['insurance'].count()
frequencyCount = count
print(frequencyCount)
probability = count/total
print(probability)

category_group_size = Data['group_size'].unique()
category_homeowner = Data['homeowner'].unique()
category_married_couple = Data['married_couple'].unique()
category_insurance = Data['insurance'].unique()

print('Unique Values of group_size: \n', category_group_size)
print('Unique Values of homeowner: \n', category_homeowner)
print('Unique Values of married_couple: \n', category_married_couple)
print('Unique Values of insurance: \n', category_insurance)

Data = pandas.read_csv('Purchase_Likelihood.csv')
count = Data.groupby('insurance').count()['group_size']
prop = count/ Data.shape[0]
grouped = pandas.DataFrame({'insurance': count.index, 
                                    'Count': count.values, 
                                    'Class Probability': prop.values})
groupsize_cross = pandas.crosstab(Data.insurance, Data.group_size, margins = False, dropna = False)
homeowner_cross = pandas.crosstab(Data.insurance, Data.homeowner, margins = False, dropna = False)
married_cross = pandas.crosstab(Data.insurance, Data.married_couple, margins = False, dropna = False)
print(groupsize_cross)
print(homeowner_cross)
print(married_cross)

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
print("Cramers statistics for group_size",cramer_group)
print("Cramers statistics for homeowner",cramer_home)
print("Cramers statistics for married_couple",cramer_married)

print("the largest association value",max(cramer_group,cramer_home,cramer_married))

def prob(x):
    prob0 = ((grouped['Count'][0] / grouped['Count'].sum()) * 
                   (groupsize_cross[x[0]][0] / groupsize_cross.loc[[0]].sum(axis=1)[0]) * 
                   (homeowner_cross[x[1]][0] / homeowner_cross.loc[[0]].sum(axis=1)[0]) * 
                   (married_cross[x[2]][0] / married_cross.loc[[0]].sum(axis=1)[0]))
    prob1 = ((grouped['Count'][1] / grouped['Count'].sum()) * 
                   (groupsize_cross[x[0]][1] / groupsize_cross.loc[[1]].sum(axis=1)[1]) * 
                   (homeowner_cross[x[1]][1] / homeowner_cross.loc[[1]].sum(axis=1)[1]) * 
                   (married_cross[x[2]][1] / married_cross.loc[[1]].sum(axis=1)[1]))
    prob2 = ((grouped['Count'][2] / grouped['Count'].sum()) * 
                   (groupsize_cross[x[0]][2] / groupsize_cross.loc[[2]].sum(axis=1)[2]) * 
                   (homeowner_cross[x[1]][2] / homeowner_cross.loc[[2]].sum(axis=1)[2]) * 
                   (married_cross[x[2]][2] / married_cross.loc[[2]].sum(axis=1)[2]))
    probs_sum = prob0 + prob1 + prob2
    probfin0 = prob0 / probs_sum
    probfin1 = prob1 / probs_sum
    probfin2 = prob2 / probs_sum

    return [probfin0, probfin1, probfin2]

unique_group = sorted(list(Data.group_size.unique()))
unique_home = sorted(list(Data.homeowner.unique()))
unique_married_couple = sorted(list(Data.married_couple.unique()))

c = list(itertools.product(unique_group, unique_home, unique_married_couple))
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
Test.to_excel("prob.xlsx")


m=[]
for i in range(len(nbp)):
    temp=nbp[i][1]/nbp[i][0]
    m.append([temp])    
print(numpy.array(m).max())

numpy.array(m).max()
print('value of combination',c[numpy.where(m == numpy.array(m).max())[0][0]])