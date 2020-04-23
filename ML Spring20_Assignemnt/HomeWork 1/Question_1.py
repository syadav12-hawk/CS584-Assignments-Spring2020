
"""
Name : Sourav Yadav 
ID : A20450418
CS583 Spring 2020
Assignment 1

"""
# Package Importer
import pandas
import math
import matplotlib.pyplot as plt
from scipy.stats import iqr



X = pandas.read_csv('C:\\Users\\soura\\OneDrive\\Desktop\\ML\\ML Spring20_Assignemnt\\HomeWork 1\\NormalSample.csv',
                       delimiter=',')

t=X[['x']]
t=t.sort_values(by=['x'])
N=int(X[['x']].count())
maxi=float(t.max())
mini=float(t.min())
qr=iqr(X[['x']])
h_ize=float(2*qr*pow(N,-1/3))
print(f"Izenman bin-width: {h_ize}")


##Histogram Estimator
#-------------------------------------------------------
h=2
print(f"The value of h is : {h}")
a=math.floor(mini)
b=math.ceil(maxi)
#no_bins=math.ceil((b-a)/h)
no_bins=int(((b-a)/h))
print(f"Number of bins : {no_bins}")



tl=t['x'].tolist() 
prob,y,z=plt.hist(tl, bins=no_bins,range=[a,b],density=True,
        zorder=5, edgecolor='k', alpha=1)
plt.xlabel('X values')
plt.ylabel('Probabilities')
plt.title('Histogram Estimator')


#Printing Co-ordinates of the histogram
print("Co-ordinates of histogram given in format: (mid point of histogram,p(mid point of histogram))")
for i in range(0,len(prob)): 
  print(f"({y[i]+h/2},{prob[i]})")





  



  
  
  
  
  


