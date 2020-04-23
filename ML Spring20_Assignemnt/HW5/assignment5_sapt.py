#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 17:30:21 2020

@author: saptarshimaiti
"""



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas
import sklearn
import sklearn.neural_network as nn
import sklearn.metrics as metrics 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


#Import Dataset
df = pd.read_csv('C:\\Users\\soura\\OneDrive\\Desktop\\ML\\ML Spring20_Assignemnt\\HW5\\SpiralWithCluster.csv')

df.head()


#-------------Question 1--------------#

##### --a--
count_all = df['SpectralCluster'].count()
count_one = df.SpectralCluster[df.SpectralCluster == 1].count()
percent_one = (count_one/count_all)*100
print("Percent of the observations have SpectralCluster equals to 1: ",percent_one)


##### --b--

pd.set_option('display.max_columns', 500)

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
    RSquare = metrics.r2_score(target, y_pred) 
    accuracy = accuracy_score(target, y_pred)
    misclassify = 1 - accuracy 
    
    return Loss, RSquare, misclassify, iterate, labels, output_activation


stats_ls = []

activation_function = ['identity', 'logistic', 'relu','tanh']
hidden_layer = 5
neurons_no = 10 

for function in activation_function: 
    for layer in range(1, hidden_layer+1):
        for neurons in range(1, neurons_no+1):
            Loss, RSquare, misclassify, iterate, labels, output_activation = neural_network(function, layer, neurons)
            stats_ls.append([function, layer, neurons,RSquare, iterate, Loss, misclassify])
stats_df = pd.DataFrame(stats_ls, columns = ['Activation function', 'Number of layers', 'Neurons per layer', 'RSquare',
                                            'Iterations performed','Loss value', 'Misclassification rate']) 


    
    
stats_df_relu = stats_df[stats_df["Activation function"] == "relu"]
stats_df_identity = stats_df[stats_df["Activation function"] == "identity"]
stats_df_logistic = stats_df[stats_df["Activation function"] == "logistic"]
stats_df_tanh = stats_df[stats_df["Activation function"] == "tanh"]

relu_frame = stats_df_relu[stats_df_relu["Loss value"] == stats_df_relu["Loss value"].min()]
identity_frame = stats_df_identity[stats_df_identity["Loss value"] == stats_df_identity["Loss value"].min()]
logistics_frame = stats_df_logistic[stats_df_logistic["Loss value"] == stats_df_logistic["Loss value"].min()]
tanh_frame = stats_df_tanh[stats_df_tanh["Loss value"] == stats_df_tanh["Loss value"].min()]

frames = [relu_frame, identity_frame, logistics_frame, tanh_frame]
result_table = pd.concat(frames)
result_table[['Activation function', 'Number of layers', 'Neurons per layer', 'Iterations performed', 'Loss value', 'Misclassification rate']]

##### --c--
print("Name of the output activation function: ",output_activation)

##### --d--
stats_df_relu = stats_df[stats_df["Activation function"] == "relu"]
stats_df_identity = stats_df[stats_df["Activation function"] == "identity"]
stats_df_logistic = stats_df[stats_df["Activation function"] == "logistic"]
stats_df_tanh = stats_df[stats_df["Activation function"] == "tanh"]

relu_frame = stats_df_relu[stats_df_relu["Loss value"] == stats_df_relu["Loss value"].min()]
identity_frame = stats_df_identity[stats_df_identity["Loss value"] == stats_df_identity["Loss value"].min()]
logistics_frame = stats_df_logistic[stats_df_logistic["Loss value"] == stats_df_logistic["Loss value"].min()]
tanh_frame = stats_df_tanh[stats_df_tanh["Loss value"] == stats_df_tanh["Loss value"].min()]

frames = [relu_frame, identity_frame, logistics_frame, tanh_frame]
result_table = pd.concat(frames)
 

min_loss = float("inf")
min_miss_rate = float("inf")
index = None
# finding activation function, number of layers, and number of neurons per layer with minimum loss and misclassification rate
for ind, row in result_table.iterrows():
    if row['Loss value'] <= min_loss and row['Misclassification rate'] <= min_miss_rate:
        index = ind
        min_loss = row['Loss value']
        min_miss_rate = row['Misclassification rate']

print(result_table.loc[index])

##### --e--
optimal_clf = nn.MLPClassifier(solver='lbfgs', learning_rate_init = 0.1, max_iter = 5000,activation = 'relu', 
                  hidden_layer_sizes= (8,)*4, random_state = 20200408)
predic = df[["x","y"]]
target = df.SpectralCluster
thisFit = optimal_clf.fit(predic, target) 
optimal_y_pred = optimal_clf.predict(predic)
optimal_clf_pred_prob = optimal_clf.predict_proba(predic)
df['NLPpredictions'] = optimal_y_pred
pred_proba = pd.DataFrame(data=optimal_clf_pred_prob,columns = ["clas0","class1"])


import seaborn as sns
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
ax.grid(b=True, which='major')
colors = ['red','blue']
for i in range(2):
    Data = df[df['NLPpredictions']==i]
    plt.scatter(Data.x,Data.y,c = colors[i],label=i)
    #plt.legend()
plt.title("Scatterplot according to Cluster Values Predicted by optimal neural network")
plt.xlabel("X Co-ordinate")
plt.ylabel("Y Co-ordinate")
plt.legend()

##### --f--
pd.set_option('float_format', '{:.10f}'.format)

threshold = count_one/count_all

pred_prob_class1 = pred_proba[pred_proba["class1"] > 0.5]["class1"]
pred_prob_class1 = pred_prob_class1.to_list()
print("Count of predicted probability Prob(SpectralCluster = 1): ",len(pred_prob_class1))
print("Mean of predicted probability Prob(SpectralCluster = 1): ",round(np.mean(pred_prob_class1),10))
print("Standard Deviation of predicted probability Prob(SpectralCluster = 1): ",round(np.std(pred_prob_class1),10))
    
#pred_prob_class1.describe()

#-------------Question 2--------------#
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import sklearn.svm as svm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

spiral_with_cluster = df

# building svm classifier
x_train = spiral_with_cluster[['x', 'y']]
y_train = spiral_with_cluster['SpectralCluster']

svm_model = svm.SVC(kernel='linear', decision_function_shape='ovr', random_state=20200408, max_iter=-1)
this_fit = svm_model.fit(x_train, y_train)

# Answer 2 a)

print('Intercept is ', this_fit.intercept_)
print('Coefficients are ', np.round(this_fit.coef_, 7))
print(
    'Separating Hyperplane Eq. is  ==> '
    f'({np.round(this_fit.intercept_[0], 7)}) '
    f'+ ({np.round(this_fit.coef_[0][0], 7)}*x_1) '
    f'+ ({np.round(this_fit.coef_[0][1], 7)}*x_2) = 𝟎')


# Answer 2 b)
y_predict_class = this_fit.predict(x_train)

# Accuracy and Misclassification rate
accuracy = metrics.accuracy_score(y_train, y_predict_class)
miss_rate = 1 - accuracy
print('Accuracy of the SVM: ', accuracy)
print('Misclassification rate of the SVM : ', miss_rate)

# Answer 2 c)

spiral_with_cluster['pred_class'] = y_predict_class

# Getting the hyperplane
xx = np.linspace(-5, 5)
yy = np.zeros((len(xx), 1))
for j in range(1):
    w = this_fit.coef_[j, :]
    a = -w[0] / w[1]
    yy[:, j] = a * xx - (this_fit.intercept_[j]) / w[1]

# Plotting the line, coordinates, and the nearest vectors to the plane
color_array = ['red', 'blue']
for i in range(2):
    x_y = spiral_with_cluster[spiral_with_cluster['pred_class'] == i]
    plt.scatter(x_y['x'], x_y['y'], c=color_array[i], label=i)
plt.plot(xx, yy[:, 0], color='black', linestyle='--')
plt.xlabel('X - coordinate')
plt.ylabel('Y - coordinate')
plt.title('SVM on 2 Segments')
plt.legend(title='Predicted Cluster', loc='best')
plt.grid(True)
plt.show()


# Answer 2 d)

def customArcTan(z):
    theta = np.where(z < 0.0, 2.0 * np.pi + z, z)
    return (theta)


# Getting radius and theta coordinates
spiral_with_cluster['radius'] = np.sqrt(spiral_with_cluster['x'] ** 2 + spiral_with_cluster['y'] ** 2)
spiral_with_cluster['theta'] = np.arctan2(spiral_with_cluster['y'], spiral_with_cluster['x']).apply(customArcTan)

# Plotting the polar coordinates
color_array = ['red', 'blue']
for i in range(2):
    x_y = spiral_with_cluster[spiral_with_cluster['SpectralCluster'] == i]
    plt.scatter(x_y['radius'], x_y['theta'], c=color_array[i], label=i)
plt.xlabel('Radius - Coordinate')
plt.ylabel('Theta - Coordinate')
plt.ylim(-1, 7)
plt.title('SVM on Two Segments')
plt.legend(title='Spectral Cluster', loc='best')
plt.grid(True)
plt.show()


# Answer 2 e)
group = np.zeros(spiral_with_cluster.shape[0])

# Creating four groups using the location of the coordinates
for index, row in spiral_with_cluster.iterrows():
    if row['radius'] < 1.5 and row['theta'] > 6:
        group[index] = 0
    elif row['radius'] < 2.5 and row['theta'] > 3:
        group[index] = 1
    elif 2.5 < row['radius'] < 3 and row['theta'] > 5.5:
        group[index] = 1
    elif row['radius'] < 2.5 and row['theta'] < 3:
        group[index] = 2
    elif 3 < row['radius'] < 4 and 3.5 < row['theta'] < 6.5:
        group[index] = 2
    elif 2.5 < row['radius'] < 3 and 2 < row['theta'] < 4:
        group[index] = 2
    elif 2.5 < row['radius'] < 3.5 and row['theta'] < 2.25:
        group[index] = 3
    elif 3.55 < row['radius'] and row['theta'] < 3.25:
        group[index] = 3

spiral_with_cluster['group'] = group

# Plotting the divided coordinates
color_array = ['red', 'blue', 'green', 'black']
for i in range(4):
    x_y = spiral_with_cluster[spiral_with_cluster['group'] == i]
    plt.scatter(x=x_y['radius'], y=x_y['theta'], c=color_array[i], label=i)
plt.xlabel('Radius - Coordinates')
plt.ylabel('Theta - Coordinates')
plt.title('SVM on 4 Segments')
plt.legend(title='Group', loc='best')
plt.grid(True)
plt.show()


# Answer 2 f)

# build SVM 0: Group 0 versus Group 1
svm_1 = svm.SVC(kernel="linear", random_state=20200408, decision_function_shape='ovr', max_iter=-1)
subset1 = spiral_with_cluster[spiral_with_cluster['group'] == 0]
subset1 = subset1.append(spiral_with_cluster[spiral_with_cluster['group'] == 1])
train_subset1 = subset1[['radius', 'theta']]
svm_1.fit(train_subset1, subset1['SpectralCluster'])

# build SVM 1: Group 1 versus Group 2
svm_2 = svm.SVC(kernel="linear", random_state=20200408, decision_function_shape='ovr', max_iter=-1)
subset2 = spiral_with_cluster[spiral_with_cluster['group'] == 1]
subset2 = subset2.append(spiral_with_cluster[spiral_with_cluster['group'] == 2])
train_subset2 = subset2[['radius', 'theta']]
svm_2.fit(train_subset2, subset2['SpectralCluster'])

# build SVM 2: Group 2 versus Group 3
svm_3 = svm.SVC(kernel="linear", random_state=20200408, decision_function_shape='ovr', max_iter=-1)
subset3 = spiral_with_cluster[spiral_with_cluster['group'] == 2]
subset3 = subset3.append(spiral_with_cluster[spiral_with_cluster['group'] == 3])
train_subset3 = subset3[['radius', 'theta']]
svm_3.fit(train_subset3, subset3['SpectralCluster'])

print(f'Separating Hyperplane Eq. for SVM 0 is   ==> ' f'({np.round(svm_1.intercept_[0] ,7)})' f' + ({np.round(svm_1.coef_[0][0], 7)}*x_1)' f' + ({np.round(svm_1.coef_[0][1], 7)}*x_2) = 𝟎')
print(f'Separating Hyperplane Eq. for SVM 1 is   ==> ' f'({np.round(svm_2.intercept_[0] ,7)})' f' + ({np.round(svm_2.coef_[0][0], 7)}*x_1)' f' + ({np.round(svm_2.coef_[0][1], 7)}*x_2) = 𝟎')
print(f'Separating Hyperplane Eq. for SVM 2 is   ==> ' f'({np.round(svm_3.intercept_[0] ,7)})' f' + ({np.round(svm_3.coef_[0][0], 7)}*x_1)' f' + ({np.round(svm_3.coef_[0][1], 7)}*x_2) = 𝟎')


# Answer 2 g)
w = svm_1.coef_[0]
a = -w[0] / w[1]
xx1 = np.linspace(1, 4)
yy1 = a * xx1 - (svm_1.intercept_[0]) / w[1]

w = svm_2.coef_[0]
a = -w[0] / w[1]
xx2 = np.linspace(1, 4)
yy2 = a * xx2 - (svm_2.intercept_[0]) / w[1]

w = svm_3.coef_[0]
a = -w[0] / w[1]
xx3 = np.linspace(1, 4)
yy3 = a * xx3 - (svm_3.intercept_[0]) / w[1]

# Plotting the polar coordinates and hyperplanes
for i in range(4):
    x_y = spiral_with_cluster[spiral_with_cluster['group'] == i]
    plt.scatter(x_y['radius'], x_y['theta'], c=color_array[i], label=i)
plt.plot(xx1, yy1, color='black', linestyle='-')
plt.plot(xx2, yy2, color='black', linestyle='-')
plt.plot(xx3, yy3, color='black', linestyle='-')
plt.xlabel('Radius - Coordinates')
plt.ylabel('Theta - Coordinates')
plt.title('SVM on 4 Segments')
plt.legend(title='Group', loc='best', )
plt.grid(True)
plt.show()


# Answer 2 h)
h1_xx1, h1_yy1 = xx1 * np.cos(yy1), xx1 * np.sin(yy1)

h2_xx2, h2_yy2 = xx2 * np.cos(yy2), xx2 * np.sin(yy2)

h3_xx3, h3_yy3 = xx3 * np.cos(yy3), xx3 * np.sin(yy3)


# Plotting the line, coordinates, and the nearest vectors to the plane
color_array = ['red', 'blue']
for i in range(2):
    x_y = spiral_with_cluster[spiral_with_cluster['SpectralCluster'] == i]
    plt.scatter(x_y['x'], x_y['y'], c=color_array[i], label=i)
plt.plot(h1_xx1, h1_yy1, color='green', linestyle='--')
plt.plot(h2_xx2, h2_yy2, color='black', linestyle='--')
plt.plot(h3_xx3, h3_yy3, color='black', linestyle='--')
plt.xlabel('X - coordinates')
plt.ylabel('Y - coordinates')
plt.title('SVM on 2 Segments')
plt.legend(title='Spectral Cluster', loc='best')
plt.grid(True)
plt.show()
