#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas
import sklearn
import sklearn.neural_network as nn
import sklearn.metrics as metrics 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[28]:


df = pd.read_csv('C:\\Users\\soura\\OneDrive\\Desktop\\ML\\ML Spring20_Assignemnt\\HW5\\SpiralWithCluster.csv')
# df.head()


# #Question1 

# In[3]:


#a)
FullC = df['SpectralCluster'].count()
OneC = df.SpectralCluster[df.SpectralCluster == 1].count()
Perc = (OneC/FullC)*100
print("Percent for value 1: ",Perc)


# In[4]:


#b)

def neural_network(function, layer, neurons):
    clf = nn.MLPClassifier(solver='lbfgs', learning_rate_init = 0.1, max_iter = 5000,activation = function, 
                  hidden_layer_sizes= (neurons,)*layer, random_state= 20200408) 
    predic = df[["x","y"]]
    target = df.SpectralCluster
    thisFit = clf.fit(predic, target) 
    y_pred = clf.predict(predic)
    
    labels = clf.classes_ 
    Loss = clf.loss_
    iterate = clf.n_iter_
    output_activation = clf.out_activation_  
    accuracy = accuracy_score(target, y_pred)
    misclassify = 1 - accuracy 
    
    return Loss, misclassify, iterate, labels, output_activation


# In[5]:


stats_ls = []

actiFn = ['identity', 'logistic', 'relu','tanh']
HLayer = 5
NNo = 10 

for function in actiFn: 
    for layer in range(1, HLayer+1):
        for neurons in range(1, NNo+1):
            Loss, misclassify, iterate, labels, output_activation = neural_network(function, layer, neurons)
            stats_ls.append([function, layer, neurons, iterate, Loss, misclassify])
stats = pd.DataFrame(stats_ls, columns = ['Activation function', 'Number of layers', 'Neurons per layer',
                                            'Iterations performed','Loss value', 'Misclassification rate']) 


# In[6]:


statsR = stats[stats["Activation function"] == "relu"]
statsI = stats[stats["Activation function"] == "identity"]
statsL = stats[stats["Activation function"] == "logistic"]
statsT = stats[stats["Activation function"] == "tanh"]

Rframe = statsR[statsR["Loss value"] == statsR["Loss value"].min()]
Iframe = statsI[statsI["Loss value"] == statsI["Loss value"].min()]
Lframe = statsL[statsL["Loss value"] == statsL["Loss value"].min()]
Tframe = statsT[statsT["Loss value"] == statsT["Loss value"].min()]

frames = [Rframe, Iframe, Lframe, Tframe]
result = pd.concat(frames)
print(result)


# In[8]:


#c) 
print("Output activation function: ",output_activation)


# In[31]:


#d)
statsR = stats[stats["Activation function"] == "relu"]
statsI = stats[stats["Activation function"] == "identity"]
statsL = stats[stats["Activation function"] == "logistic"]
statsT = stats[stats["Activation function"] == "tanh"]

Rframe = statsR[statsR["Loss value"] == statsR["Loss value"].min()]
Iframe = statsI[statsI["Loss value"] == statsI["Loss value"].min()]
Lframe = statsL[statsL["Loss value"] == statsL["Loss value"].min()]
Tframe = statsT[statsT["Loss value"] == statsT["Loss value"].min()]

frames = [Rframe, Iframe, Lframe, Tframe]
result = pd.concat(frames)
print(result)


# In[38]:


abc = result.iloc[0]
print("Activation function with lowest loss and misclassification rate:\n\n",abc)


# In[10]:


optimal_clf = nn.MLPClassifier(solver='lbfgs', learning_rate_init = 0.1, max_iter = 5000,activation = 'relu', 
                  hidden_layer_sizes= (10,)*4, random_state = 20200408)
predic = df[["x","y"]]
target = df.SpectralCluster
thisFit = optimal_clf.fit(predic, target) 
optimal_y_pred = optimal_clf.predict(predic)
optimal_clf_pred_prob = optimal_clf.predict_proba(predic)
df['NLPpredictions'] = optimal_y_pred
pred_proba = pd.DataFrame(data=optimal_clf_pred_prob,columns = ["clas0","class1"])


# In[11]:


#e)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
ax.grid(b=True, which='major')
colors = ['red','blue']
for i in range(2):
    Data = df[df['NLPpredictions']==i]
    plt.scatter(Data.x,Data.y,c = colors[i],label=i)
plt.title("Scatterplot according to Cluster Values Predicted by optimal neural network")
plt.xlabel("X Co-ordinate")
plt.ylabel("Y Co-ordinate")
plt.legend()


# In[42]:


#f) 
# pred_prob_class1 = pred_proba[pred_proba["class1"] > 0.5]["class1"]
# pred_prob_class1 = pred_prob_class1.to_list()
# print("Count of predicted probability Prob(SpectralCluster = 1): ",len(pred_prob_class1))
# print("Mean of predicted probability Prob(SpectralCluster = 1): ",round(np.mean(pred_prob_class1),10))
# print("Standard Deviation of predicted probability Prob(SpectralCluster = 1): ",round(np.std(pred_prob_class1),10))


y_predProb = optimal_clf.predict_proba(df[["x","y"]])
df['y_predProb']=y_predProb[:,1]
group=df.groupby('SpectralCluster')
standard_deviation=group.get_group(1).describe()['y_predProb']['std']
mean=group.get_group(1).describe()['y_predProb']['mean']
count=group.get_group(1).describe()['y_predProb']['count']
print("Count: ",count,"\n")
print("Mean: ",mean,"\n")
print("Standard deviation: ",standard_deviation,"\n")


# In[13]:


# Question 2


# In[14]:


#a) 
# It is solved below question 2(c)


# In[15]:


#b) 

from sklearn.svm import SVC

predic = df[["x","y"]]
target = df.SpectralCluster
svm_clf = SVC(kernel = "linear", random_state=20200408, decision_function_shape='ovr',max_iter=-1,probability = True)
svm_clf.fit(predic,target)

svm_pred = svm_clf.predict(predic)

svm_accuracy = accuracy_score(target, svm_pred)
svm_missclassification = 1- svm_accuracy
print("The miscalssification rate is: ", svm_missclassification)


# In[16]:


df["SVM_pred"] = svm_pred


# In[17]:


#c)

coeff = svm_clf.coef_[0]
a = -coeff[0] / coeff[1]
xx = np.linspace(-5, 5)
yy = a * xx - (svm_clf.intercept_[0]) / coeff[1]

carray=['red','blue']
fig, ax = plt.subplots(1, 1)
ax.grid(b=True, which='major')

plt.plot(xx, yy, 'k--')


for i in range(2):
    Data = df[df["SVM_pred"]==i]
    plt.scatter(Data.x,Data.y,label = (i),c = carray[i])
plt.legend()
plt.title("Scatterplot of Cluster Values")
plt.xlabel("X Co-ordinate")
plt.ylabel("Y Co-ordinate")


# In[18]:


#a) 
print ('Equation of the seperating hyperplane is')
print (round(svm_clf.intercept_[0],7), " + (", round(coeff[0],7), ") X +(" ,round(coeff[1],7),") Y = 0")


# In[19]:


def customArcTan (z):
    theta = np.where(z < 0.0, 2.0*np.pi+z, z)
    return (theta)

trainData = pd.DataFrame(columns = ["radius","theta"])
trainData['radius'] = np.sqrt(df['x']**2 + df['y']**2)
trainData['theta'] = (np.arctan2(df['y'], df['x'])).apply(customArcTan)
trainData['class']=df["SpectralCluster"]


# In[20]:


#d) 
colour = ['red','blue']
for i in range(2):
    Data = trainData[trainData["class"]==i]
    plt.scatter(Data.radius,Data.theta,label = (i),c = colour[i])
    
plt.title("Scatterplot of Co-ordinates")
plt.xlabel("Radius")
plt.ylabel('Theta Co-ordinate')
plt.legend()
plt.grid()


# In[21]:


#e)
x = trainData["radius"]
y = trainData['theta'].apply(customArcTan)
svm_df = pd.DataFrame(columns = ['Radius','Theta'])
svm_df['Radius'] = x
svm_df['Theta'] = y

group = []

for i in range(len(x)):
    if x[i] < 1.5 and y[i]>6:
        group.append(0)
        
    elif x[i] < 2.5 and y[i]>3 :
        group.append(1)
    
    elif 2.75 > x[i]>2.5 and y[i]>5:
        group.append(1)
        
    elif 2.5<x[i]<3 and 2<y[i]<4:
        group.append(2)   
    
    elif x[i]> 2.5 and y[i]<3.1:
        group.append(3)
        
    elif x[i] < 4:
        group.append(2)
        

svm_df['Group'] = group
colors = ['red','blue','green','black']
for i in range(4):
    Data = svm_df[svm_df.Group == i]
    plt.scatter(Data.Radius,Data.Theta,c = colors[i],label=i)
plt.grid()
plt.title("Scatterplot with four Groups")
plt.xlabel("Radius")
plt.ylabel('Theta Co-ordinate')
plt.legend()


# In[24]:


#SVM to classify class 0 and class 1
svm_1 = SVC(kernel = "linear", random_state=20200408, decision_function_shape='ovr',max_iter=-1,probability = True)
x = svm_df[svm_df['Group'] == 0]
x = x.append(svm_df[svm_df['Group'] == 1])
td = x[['Radius','Theta']]
svm_1.fit(td,x.Group)

coeff = svm_1.coef_[0]
a = -coeff[0] / coeff[1]
xx = np.linspace(1, 2)
yy = a * xx - (svm_1.intercept_[0])/coeff[1] 

print ('Equation of the hypercurve for SVM 0 is')
print (svm_1.intercept_[0], " + (", coeff[0], ") X +(" ,coeff[1],") Y = 0")

h0_xx = xx * np.cos(yy[:])
h0_yy = xx * np.sin(yy[:])

carray=['red','blue','green','black']
fig, ax = plt.subplots(1, 1)
ax.grid(b=True, which='major')

#Plot the hyperplane
plt.plot(xx, yy, 'k--')

#SVM to classify class 1 and class 2
svm_1 = SVC(kernel = "linear", random_state=20200408, decision_function_shape='ovr',max_iter=-1,probability = True)
x = svm_df[svm_df['Group'] == 1]
x = x.append(svm_df[svm_df['Group'] == 2])
td = x[['Radius','Theta']]
svm_1.fit(td,x.Group)

coeff = svm_1.coef_[0]
a = -coeff[0] / coeff[1]
xx = np.linspace(1, 4)
yy = a * xx - (svm_1.intercept_[0])/coeff[1] 
print ('Equation of the hypercurve for SVM 1 is')
print (svm_1.intercept_[0], " + (", coeff[0], ") X +(" ,coeff[1],") Y = 0")

h1_xx = xx * np.cos(yy[:])
h1_yy = xx * np.sin(yy[:])

#Plot ther hyperplane
plt.plot(xx, yy, 'k--')

#SVM to. classify class 2 and class 3
svm_1 = SVC(kernel = "linear", random_state=20200408, decision_function_shape='ovr',max_iter=-1,probability = True)
x = svm_df[svm_df['Group'] == 2]
x = x.append(svm_df[svm_df['Group'] == 3])
td = x[['Radius','Theta']]
svm_1.fit(td,x.Group)

coeff = svm_1.coef_[0]
a = -coeff[0] / coeff[1]
xx = np.linspace(2, 4.5)
yy = a * xx - (svm_1.intercept_[0])/coeff[1] 
print ('Equation of the hypercurve for SVM 2 is')
print (svm_1.intercept_[0], " + (", coeff[0], ") X +(" ,coeff[1],") Y = 0")

h2_xx = xx * np.cos(yy[:])
h2_yy = xx * np.sin(yy[:])


#Plot ther hyperplane
plt.plot(xx, yy, 'k--')

for i in range(4):
    Data = svm_df[svm_df.Group == i]
    plt.scatter(Data.Radius,Data.Theta,c = carray[i],label=i)
plt.xlabel("Radius")
plt.ylabel("Theta Co-Ordinate")
plt.title("Theta-coordinate against the Radius-coordinate in a scatterplot seperated by 3 hyperplanes")
plt.legend()


# In[26]:


carray=['red','blue']
fig, ax = plt.subplots(1, 1)
ax.grid(b=True, which='major')

plt.plot(h0_xx, h0_yy, 'k--')
plt.plot(h1_xx, h1_yy, 'k--')
plt.plot(h2_xx, h2_yy, 'k--')

for i in range(2):
    Data = df[df["SpectralCluster"]==i]
    plt.scatter(Data.x,Data.y,label = (i),c = carray[i])
plt.legend()
plt.title("Scatterplot of the cartesian co-ordinates with hypercurves")
plt.xlabel("X Co-ordinate")
plt.ylabel("Y Co-ordinate")


# In[27]:


carray=['red','blue']
fig, ax = plt.subplots(1, 1)
ax.grid(b=True, which='major')


plt.plot(h1_xx, h1_yy, 'k--')
plt.plot(h2_xx, h2_yy, 'k--')

for i in range(2):
    Data = df[df["SpectralCluster"]==i]
    plt.scatter(Data.x,Data.y,label = (i),c = carray[i])
plt.legend()
plt.title("Scatterplot of the cartesian co-ordinates with hypercurves")
plt.xlabel("X Co-ordinate")
plt.ylabel("Y Co-ordinate")

