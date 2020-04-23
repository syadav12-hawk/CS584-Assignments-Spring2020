"""
Name : Sourav Yadav 
ID : A20450418
CS583 Spring 2020
Assignment 1

"""

import pandas
import matplotlib.pyplot as plt
from scipy.stats import iqr



X = pandas.read_csv('C:\\Users\\soura\\OneDrive\\Desktop\\ML\\ML Spring20_Assignemnt\\HomeWork 1\\NormalSample.csv',
                       delimiter=',')
t=X[['x']]
t=t.sort_values(by=['x'])
maxi=float(t.max())
mini=float(t.min())
qr=iqr(X[['x']])
tl=t['x'].tolist()

y=plt.boxplot(tl)
q1,median,q3=list(X.x.quantile([0.25,0.5,0.75]))

#Calculating five number summary of All X values
print("Five number summary of x:")
print(f"Q1 :{q1}")
print(f"Mediean :{median}")
print(f"Q3 :{q3}")
print(f"Maximum :{maxi}")
print(f"Minimum :{mini}")
upw=q3+qr*1.5
loww=q1-qr*1.5
print(f"1.5 IQR whikers :{upw},{loww}\n")


#Extarcting Group1 and Group0 Data from Original dataframe
group1=X[X['group'] == 1]
group0=X[X['group']==0]
grpls1=group1['x'].tolist()
grpls0=group0['x'].tolist()


#Calculating five number summary of Group1
grp1_q1,grp1_median,grp1_q3=list(group1.x.quantile([0.25,0.5,0.75]))
print("Five number summary of x of group1:")
print(f"Q1 :{grp1_q1}")
print(f"Median :{grp1_median}")
print(f"Q3 :{grp1_q3}")
print(f"Maximum :{float(group1[['x']].max())}")
print(f"Minimum :{float(group1[['x']].min())}")
upw_grp1=grp1_q3+iqr(group1[['x']])*1.5
loww_grp1=grp1_q1-iqr(group1[['x']])*1.5
print(f"1.5 IQR whiskers :{upw_grp1},{loww_grp1}\n")


#Calculating five number summary of Group2
grp0_q1,grp0_median,grp0_q3=list(group0.x.quantile([0.25,0.5,0.75]))
print("Five number summary of x of group0:")
print(f"Q1 :{grp0_q1}")
print(f"Median :{grp0_median}")
print(f"Q3 :{grp0_q3}")
print(f"Maximum :{float(group0[['x']].max())}")
print(f"Minimum :{float(group0[['x']].min())}")
#Upper and Lower Whiskers
upw_grp0=grp0_q3+iqr(group0[['x']])*1.5
loww_grp0=grp0_q1-iqr(group0[['x']])*1.5
print(f"1.5 IQR whiskers:{upw_grp0},{loww_grp0}\n")


#Plotting three group data in same frame
all_data=[tl,group1['x'].tolist(),group0['x'].tolist()]
plt.boxplot(all_data,labels=['All X','Group1','Group0'])


#Function for outlier Detection
def outlier_detect(in_list,upper_wishker,lower_wishker):
  outerliers=[]
  for i in in_list:
    if i >upper_wishker or i<lower_wishker:
      outerliers.append(i)
  return outerliers

#Outliers of All X -group
allx_outlier=outlier_detect(tl, upw, loww)
print(f"Outliers of all X values :{allx_outlier}\n")

#Outliers of Group1
grp1_outlier=outlier_detect(grpls1,upw_grp1,loww_grp1)
print(f"Outliers of Group 1  :{grp1_outlier}\n")

#Outliers of Group0
grp0_outlier=outlier_detect(grpls0,upw_grp0,loww_grp0)
print(f"Outliers of Group 0  :{grp0_outlier}\n")

      

