#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name : Sourav Yadav
AID : A20450418

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
dataframe = pd.read_csv('C:\\Users\\soura\\OneDrive\\Desktop\\ML\\ML Spring20_Assignemnt\\HW5\\SpiralWithCluster.csv')

dataframe.head()

#-----------------------------------1.a------------------------------------
no_of_obs = dataframe.shape[0]
no_of_obs_spe_clu_1 = dataframe[dataframe['SpectralCluster'] == 1].shape[0]
per_obs_spe_clu_1 = (no_of_obs_spe_clu_1) / no_of_obs
print('Percentage of Obeserved Spectral Cluster = 1: ', 100 * per_obs_spe_clu_1)

#-----------------------------------1.b------------------------------------
act_funs = ['identity', 'logistic', 'relu', 'tanh']
num_of_layers = range(1, 6, 1)
n_neurons_per_layer = range(1, 11, 1)
result = pd.DataFrame(columns=['Index', 'ActivationFunction', 'nLayers', 'nNeuronsPerLayer', 'nIterations', 'Loss', 'MisclassificationRate'])
index = 0
# looping for all possible networks for given configuration
for act_fun in act_funs:
    min_loss = float("inf")
    min_miss_rate = float("inf")
    n_lyrs = -1
    n_npl = -1
    niter = -1
    for no_lyrs in num_of_layers:
        for no_npl in n_neurons_per_layer:
            # building the neural network
            nn_obj = nn.MLPClassifier(hidden_layer_sizes=(no_npl,) * no_lyrs, activation=act_fun, verbose=False, solver='lbfgs', learning_rate_init=0.1, max_iter=5000, random_state=20200408)
            this_fit = nn_obj.fit(dataframe[['x', 'y']], dataframe[['SpectralCluster']])
            y_pred = nn_obj.predict_proba(dataframe[['x', 'y']])
            n_iter = nn_obj.n_iter_
            loss = nn_obj.loss_

            target_y = dataframe[['SpectralCluster']]
            no_targets = target_y.shape[0]

            # to determine the predicted class
            pred_y = np.empty_like(target_y)
            threshold = per_obs_spe_clu_1
            for i in range(no_targets):
                if y_pred[i][0] >= threshold:
                    pred_y[i] = 0
                else:
                    pred_y[i] = 1

            # calculating accuracy and misclassification rate
            accuracy = metrics.accuracy_score(target_y, pred_y)
            miss_rate = 1 - accuracy

            # finding neural network with minimum loss and misclassification rate
            if loss <= min_loss and miss_rate <= min_miss_rate:
                min_loss = loss
                min_miss_rate = miss_rate
                n_lyrs = no_lyrs
                n_npl = no_npl
                niter = n_iter

    result = result.append(pd.DataFrame([[index, act_fun, n_lyrs, n_npl, niter, min_loss, min_miss_rate]], columns=['Index', 'ActivationFunction', 'nLayers', 'nNeuronsPerLayer', 'nIterations', 'Loss', 'MisclassificationRate']))
    index += 1
result = result.set_index('Index')
pd.set_option('display.max_columns', 10)
print(result)

#-----------------------------------1.c------------------------------------
print('Activation funciton: ', nn_obj.out_activation_)

#-----------------------------------1.d------------------------------------
min_loss = float("inf")
min_miss_rate = float("inf")
index = None
# finding activation function, number of layers, and number of neurons per layer with minimum loss and misclassification rate
for ind, row in result.iterrows():
    if row['Loss'] <= min_loss and row['MisclassificationRate'] <= min_miss_rate:
        index = ind
        min_loss = row['Loss']
        min_miss_rate = row['MisclassificationRate']

print(result.loc[index])


# building neural network with minimum loss and misclassification rate
nn_obj = nn.MLPClassifier(hidden_layer_sizes=(result.loc[index]['nNeuronsPerLayer'],) * result.loc[index]['nLayers'], activation=result.loc[index]['ActivationFunction'], verbose=False, solver='lbfgs', learning_rate_init=0.1, max_iter=5000, random_state=20200408)
this_fit = nn_obj.fit(dataframe[['x', 'y']], dataframe[['SpectralCluster']])
y_pred = nn_obj.predict_proba(dataframe[['x', 'y']])

target_y = dataframe[['SpectralCluster']]
no_targets = target_y.shape[0]

# determining the predicted class
pred_y = np.empty_like(target_y)
threshold = per_obs_spe_clu_1
for i in range(no_targets):
    if y_pred[i][0] >= threshold:
        pred_y[i] = 0
    else:
        pred_y[i] = 1

dataframe['y_pred_0'] = y_pred[:, 0]
dataframe['y_pred_1'] = y_pred[:, 1]
dataframe['class'] = pred_y

target_y = dataframe[['SpectralCluster']]
accuracy = metrics.accuracy_score(target_y, pred_y)
miss_rate = 1 - accuracy

print('\n Accuracy: ', 100 * accuracy)
print('Misclassification rate: ', miss_rate)





#-----------------------------------1.e------------------------------------
color_array = ['red', 'blue']
for i in range(2):
    x_y = dataframe[dataframe['class'] == i]
    plt.scatter(x_y['x'], x_y['y'], c=color_array[i], label=i)
plt.xlabel('x - coordinate')
plt.ylabel('y - coordinate')
plt.title('Scatter Plot of Spiral Cluster Coordinates')
plt.legend(title='Predicted_Cluster', loc='best')
plt.grid(True)
plt.show()

#-----------------------------------1.f------------------------------------
pd.set_option('float_format', '{:.10f}'.format)
print(dataframe[dataframe['class'] == 1]['y_pred_1'].describe())

#-------------------------------------Question 2--------------------------#
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import sklearn.svm as svm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

spiral_with_cluster = dataframe

# building svm classifier
x_train = spiral_with_cluster[['x', 'y']]
y_train = spiral_with_cluster['SpectralCluster']

svm_model = svm.SVC(kernel='linear', decision_function_shape='ovr', random_state=20200408, max_iter=-1)
this_fit = svm_model.fit(x_train, y_train)

#-----------------------------------2.a------------------------------------

print('Intercept is ', this_fit.intercept_)
print('Coefficients are ', np.round(this_fit.coef_, 7))
print(
    'Separating Hyperplane Eq. is  ==> '
    f'({np.round(this_fit.intercept_[0], 7)}) '
    f'+ ({np.round(this_fit.coef_[0][0], 7)}*x_1) '
    f'+ ({np.round(this_fit.coef_[0][1], 7)}*x_2) = 𝟎')


#-----------------------------------2.b------------------------------------
y_predict_class = this_fit.predict(x_train)

# Accuracy and Misclassification rate
accuracy = metrics.accuracy_score(y_train, y_predict_class)
miss_rate = 1 - accuracy
print('Accuracy of the SVM: ', accuracy)
print('Misclassification rate of the SVM : ', miss_rate)

#-----------------------------------2.c------------------------------------

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


#-----------------------------------2.d------------------------------------

def custArcTan(z):
    theta = np.where(z < 0.0, 2.0 * np.pi + z, z)
    return (theta)


# Getting radius and theta coordinates
spiral_with_cluster['radius'] = np.sqrt(spiral_with_cluster['x'] ** 2 + spiral_with_cluster['y'] ** 2)
spiral_with_cluster['theta'] = np.arctan2(spiral_with_cluster['y'], spiral_with_cluster['x']).apply(custArcTan)

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


#-----------------------------------2.c------------------------------------
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


#-----------------------------------2.f------------------------------------

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


#-----------------------------------2.g------------------------------------
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


#-----------------------------------2.h------------------------------------
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
