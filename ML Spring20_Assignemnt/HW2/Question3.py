# -*- coding: utf-8 -*-
"""
Name : Sourav Yadav 
ID : A20450418
CS584-04 Spring 2020
Assignment 2

"""
import math
import matplotlib.pyplot as plt
import numpy
import pandas


import numpy.linalg as linalg
import sklearn.cluster as cluster
import sklearn.neighbors as neighbors


fc_Spiral = pandas.read_csv('C:\\Users\\soura\\OneDrive\\Desktop\\ML\\ML Spring20_Assignemnt\\HW2\\FourCircle.csv',
                         delimiter=',')

nObs = fc_Spiral.shape[0]


#-------------------------------------------PartA-------------------------------------
plt.scatter(fc_Spiral['x'], fc_Spiral['y'], c = fc_Spiral['ring'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

#--------------------------------------Part B-----------------------------------------

trainData = fc_Spiral[['x','y']]
kmeans = cluster.KMeans(n_clusters=4, random_state=0).fit(trainData)

print("Cluster Centroids = \n", kmeans.cluster_centers_)

fc_Spiral['KMeanCluster'] = kmeans.labels_

plt.scatter(fc_Spiral['x'], fc_Spiral['y'], c = fc_Spiral['KMeanCluster'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

#-------------------------------------Part C-------------------------------------------
# Fourteen nearest neighbors
kNNSpec = neighbors.NearestNeighbors(n_neighbors = 10, algorithm = 'brute', metric = 'euclidean')
nbrs = kNNSpec.fit(trainData)
d3, i3 = nbrs.kneighbors(trainData)

# Retrieve the distances among the observations
distObject = neighbors.DistanceMetric.get_metric('euclidean')
distances = distObject.pairwise(trainData)

# Create the Adjacency matrix
Adjacency_fc = numpy.zeros((nObs, nObs))
for i in range(nObs):
    for j in i3[i]:
        Adjacency_fc[i,j] = math.exp(- (distances[i][j])**2 )

# Make the Adjacency matrix symmetric
Adjacency_fc = 0.5 * (Adjacency_fc + Adjacency_fc.transpose())

# Create the Degree matrix
Degree_fc = numpy.zeros((nObs, nObs))
for i in range(nObs):
    sum = 0
    for j in range(nObs):
        sum += Adjacency_fc[i,j]
    Degree_fc[i,i] = sum

# Create the Laplacian matrix        
Lmatrix_fc = Degree_fc - Adjacency_fc

# Obtain the eigenvalues and the eigenvectors of the Laplacian matrix
evals, evecs = linalg.eigh(Lmatrix_fc)

# Series plot of the smallest five eigenvalues to determine the number of clusters
sequence = numpy.arange(1,10,1) 
plt.plot(sequence, evals[0:9,], marker = "o")
plt.xlabel('Sequence')
plt.ylabel('Eigenvalue')
plt.xticks(sequence)
plt.grid("both")
plt.show()

# Series plot of the smallest twenty eigenvalues to determine the number of neighbors
sequence = numpy.arange(1,21,1) 
plt.plot(sequence, evals[0:20,], marker = "o")
plt.xlabel('Sequence')
plt.ylabel('Eigenvalue')
plt.grid("both")
plt.xticks(sequence)
plt.show()

"""
# Inspect the values of the selected eigenvectors 
for j in range(10):
    print('Eigenvalue: ', j)
    print('              Mean = ', numpy.mean(evecs[:,j]))
    print('Standard Deviation = ', numpy.std(evecs[:,j]))
    print('  Coeff. Variation = ', scipy.stats.variation(evecs[:,j]))

"""



#------------------------------------Part-D----------------------------------------------------
Z1 = evals[0:4]
print(f"Values of the “zero” eigenvalues in scientific notation: {Z1}")

print("Adajacency Matrix")
print(Adjacency_fc)

print("Degree Matrix")
print(Degree_fc)

print("Laplacian matrix")
print(Lmatrix_fc)

#--------------------------------------Part E----------------------------------------------
Z = evecs[:,0:4]

# Perform 2-cluster K-mean on the first two eigenvectors
kmeans_spectral = cluster.KMeans(n_clusters = 4, random_state = 0).fit(Z)
fc_Spiral['SpectralCluster'] = kmeans_spectral.labels_

plt.scatter(fc_Spiral['x'], fc_Spiral['y'], c = fc_Spiral['SpectralCluster'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()