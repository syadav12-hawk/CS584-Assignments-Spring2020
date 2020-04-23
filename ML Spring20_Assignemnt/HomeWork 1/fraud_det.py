
"""
Name : Sourav Yadav 
ID : A20450418
CS583 Spring 2020
Assignment 1

"""


import pandas
import numpy 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors

X = pandas.read_csv('C:\\Users\\soura\\OneDrive\\Desktop\\ML\\ML Spring20_Assignemnt\\HomeWork 1\\Fraud.csv',
                       delimiter=',')
fraud=X[X['FRAUD']==1]
not_fraud=X[X['FRAUD']==0]

#Counts of Fraud, Not-Fraud and Total
fraud_count=int(fraud['FRAUD'].count())
not_fraud_count=int(not_fraud['FRAUD'].count())
total_count=int(X['FRAUD'].count())
print("Percentage of investigations found fraudulent: {:.4f}".format((fraud_count/total_count)*100))

#Getting Interval Variables
interval_vars=list(X.columns)[2:]

#Printing Interval Variables
for i in interval_vars:
  print(i)
  t=[fraud[i].tolist(),not_fraud[i].tolist()]
  plt.boxplot(t,vert=False,labels=['Fraud','Not Fraud'])
  plt.xlabel(i)
  plt.show()
  plt.close()
  

#Orthonormalization
#Taking Input data as matrix
x=numpy.matrix(X[interval_vars])
xtx=x.transpose()*x
  
#Eigen Decomposition
evals, evecs = numpy.linalg.eigh(xtx)
  
#Transformation Matrix
transf = evecs * numpy.linalg.inv(numpy.sqrt(abs(numpy.diagflat(evals))));
  
#Transformed X
transf_x = x * transf

print("Transformation Matrix : {}".format(transf))
print(f"Transformed X : {transf_x}")

#Counting the number of dimensions used
count=0
for j in evals:
  if j >1:
    count+=1
print(f"The number of dimensions used {count}")


#Expecting Identity Matrix
xtx = transf_x.transpose() * transf_x;
print("Expecting an Identity Matrix by multiplying tranpose of transformed X matrix  with itself = \n", xtx)
exp_one=numpy.linalg.norm(transf_x,2)
print(f"Norm of the Transformed matrix: {exp_one}")


#Training KNN algorithm
knn=KNeighborsClassifier(n_neighbors=5 , algorithm = 'brute', metric = 'euclidean')
target=numpy.transpose(X['FRAUD'].tolist())
knn.fit(transf_x,target)
print(f"Score returned by Score Function :{knn.score(transf_x,target)}")


#Calculating Five Neighbors
sample_mat=numpy.matrix([[7500,15,3,127,2,2]])
trans_sample=sample_mat*transf
neigh = NearestNeighbors(n_neighbors=5)
neigh.fit(transf_x,target)
five_nn_of=neigh.kneighbors(trans_sample,return_distance=False)
print(five_nn_of)
#Priting 5 nearest neighbors of given point. 
for i in  five_nn_of:
  pandas.set_option('display.max_columns', None)
  print(X.iloc[i])

#Predicated probability of the Given Point
pred_proba=knn.predict_proba(trans_sample)
print(f"Predicated Probabilty of Fradulent: {pred_proba[0][1]}")