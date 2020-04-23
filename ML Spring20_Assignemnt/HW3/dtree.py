
"""
Created on Fri Feb 28 19:58:39 2020

@author: soura
"""


import numpy 
import pandas
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import sys
from itertools import combinations
import math
global depth

global node_split
flag=0
depth=0
node_split=[]


claims = pandas.read_csv('C:\\Users\\soura\\OneDrive\\Desktop\\ML\\ML Spring20_Assignemnt\\HW3\\claim_history.csv',
                            delimiter=',')

claim_data = claims[['CAR_TYPE', 'OCCUPATION', 'EDUCATION']]
claim_target=claims[['CAR_USE']]
print("\n--------------------------------Question1--------------------------------")
#X_train, X_test, y_train, y_test = train_test_split(claim_data, claim_target, test_size=0.25, random_state=42)
X_train, X_test, Y_train, Y_test = train_test_split(claim_data, claim_target, test_size = 0.25, random_state = 60616, stratify = claim_target)

print("\n-------------------------Part A--------------------------")
print("Frequency Table of Target Varibale in Train")
print(Y_train['CAR_USE'].value_counts())
print(Y_train['CAR_USE'].value_counts(normalize=True))


print("\n-------------------------Part B--------------------------")
print("Frequency Table of Target Varibale in Test")
print(Y_test['CAR_USE'].value_counts())
print(Y_test['CAR_USE'].value_counts(normalize=True))

print("\n-------------------------Part C--------------------------")
#print(Y_train['CAR_USE'].value_counts()[1])
#print(claim_target['CAR_USE'].value_counts()[1])
print("The probability that an observation is in the Training partition given that CAR_USE = Commercial ::")
print(Y_train['CAR_USE'].value_counts()[1]/claim_target['CAR_USE'].value_counts()[1])


print("\n-------------------------Part D--------------------------")
print("The probability that an observation is in the Training partition given that CAR_USE = Commercial ::")
print(Y_test['CAR_USE'].value_counts()[0]/claim_target['CAR_USE'].value_counts()[0])

print("\n---------------------------------Question2--------------------------------")


def calEntropy(p1):
    e=0
    for i in p1:
        if i==0:
            continue            
        e+=(-i*math.log2(i))        
    return e


result_XY = pandas.concat([X_train, Y_train], axis=1, sort=False)

def calProba(x,y):
    normf=x+y
    return (x/normf,y/normf)



def GenerateTree(result_XY):
    global depth  
    global node_split
    if depth>2 :
      
        #Child node
        return    
    temp=[]
    highest=0

    #Calculate Entropy of the Node
    xt3=result_XY['CAR_USE'].value_counts()
    xt3.sort_index(inplace=True)
    tt3=pandas.Series([0,0],index=['Commercial', 'Private'])  
    for i in xt3.index:
        tt3[i]=xt3[i]  

    priv_p=tt3['Private']/tt3.sum()
    com_p=tt3['Commercial']/tt3.sum()
    parent_entropy=calEntropy([priv_p,com_p])
    
    re1,best_split1=selectOrdinSplit(result_XY,'EDUCATION',parent_entropy)
    temp.append((re1,best_split1,'EDUCATION'))

    re2,best_split2=selectNominSplit(result_XY,'OCCUPATION',parent_entropy)
    temp.append((re2,best_split2,'OCCUPATION'))
            
    re3,best_split3=selectNominSplit(result_XY,'CAR_TYPE',parent_entropy)
    temp.append((re3,best_split3,'CAR_TYPE'))
    #print(temp)       
    for i in range(3):        
        if temp[i][0]>highest:
            highest=temp[i][0]
            bs=temp[i][1]
            col_name=temp[i][2]
        else:
            continue
    node_split.append(bs)
    print(f"-----------Node at Depth {depth} -------------------")
    print(f"Split Criterion:")
    print(f"Predicator Name : {col_name}")
    print("Left Node ",bs[0])
    left=result_XY[result_XY[col_name].isin(bs[0])]
    t1=left["CAR_USE"].value_counts()
    t1.sort_index(inplace=True)
    print("Left Node Traget Values:")
    print(t1)
    c,p=calProba(t1[0],t1[1])
    print("Left Node Probabilities:")
    print(f"Commercial:{c}\n Private: {p}")
    print("Right Node",bs[1])
    right=result_XY[result_XY[col_name].isin(bs[1])]
    t2=right["CAR_USE"].value_counts()
    t2.sort_index(inplace=True)
    print("Right Node Traget Values:")
    print(t2)
    c1,p1=calProba(t2[0],t2[1])
    print("Right Node Probabilities:")
    print(f"Commercial:{c1}\n Private: {p1}")
    print(f"Entropy: {parent_entropy}")
    print(f"Split Entropy :{parent_entropy-highest}")
    #print(f"Private Values: {tt3['Private']}")
    #print(f"Commercial Values: {tt3['Commercial']}")
    if tt3['Commercial']>tt3['Private']:
        print("Predicted Class is Commercial")
    else:
        print("Predicted Class is Private")
    left=result_XY[result_XY[col_name].isin(bs[0])]
    right=result_XY[result_XY[col_name].isin(bs[1])]
    #print(left)
    #print(right)
    depth+=1
    return GenerateTree(left),GenerateTree(right)
        
"""
parent_entropy1=0.8610228634632935
#xvz=X_train.loc[:,['OCCUPATION']]
#cart1=xvz[['OCCUPATION']][result_XY['OCCUPATION'].isin(['Blue Collar', 'Student', 'Unknown'])]
#cart=cart1['OCCUPATION'].unique()           
cart=X_train['EDUCATION'].unique()
#cart_temp=cart.tolist()
cart_temp=['Below High School','High School', 'Bachelors', 'Masters', 'Doctors']

new=[]
for j in cart_temp:
    for i in cart:
        if j==i:
            new.append(j)
            
            
left=result_XY[result_XY['OCCUPATION'].isin(['Blue Collar', 'Student', 'Unknown'])]
right=result_XY[result_XY['OCCUPATION'].isin(['Lawyer', 'Clerical', 'Professional', 'Home Maker', 'Manager', 'Doctor'])]

#cart1=xvz[['OCCUPATION']][result_XY['OCCUPATION'].isin(['Blue Collar', 'Student', 'Unknown'])]
cart=right['CAR_TYPE'].unique()    
cart_temp=cart.tolist()
#cart_temp=['Below High School','High School', 'Bachelors', 'Masters', 'Doctors']
#print(cart_temp)
best_split=[0,0]
minimum=0
for i in range(len(cart)-1):
   for j in combinations(cart,i+1):
       #print(j)  
       split1=list(j)
#       split1=cart_temp[:i+1]
       #print(split1)
       split2=list(set(cart_temp)-set(split1))
#       split2=get_split(cart_temp,split1)
#       split2=cart[cart!=split1]
       #print(split2)
#          calSplitEntropy(k,temp)
       t2=right[['CAR_TYPE','CAR_USE']][right['CAR_TYPE'].isin(split1)]
#       t2=result_XY[['EDUCATION','CAR_USE']][result_XY['EDUCATION'].isin(split1)]
#           df[df['col1'].isin(['a', 'c', 'h'])]
       t3=right[['CAR_TYPE','CAR_USE']][right['CAR_TYPE'].isin(split2)]
#       t3=result_XY[['EDUCATION','CAR_USE']][result_XY['EDUCATION'].isin(split2)]
       xt1=t2['CAR_USE'].value_counts()
       xt2=t3['CAR_USE'].value_counts()
       print(xt1)
       print(xt2)
       #print(t2['CAR_USE'].value_counts())
       #print(xt)
       xt1.sort_index(inplace=True)
       xt2.sort_index(inplace=True)
       tt1=pandas.Series([0,0],index=['Commercial', 'Private'])
       tt2=pandas.Series([0,0],index=['Commercial', 'Private'])
       #tt1[0]=xt1[0]
       for i in range(len(xt1.index)):
           tt1[i]=xt1[i]
           
       for j in range(len(xt2.index)):
           tt2[j]=xt2[j]
#       try:               
#           tt1[1]=xt1[1]
#       except:
#           tt1[1]=0
       #tt.replace(tt,xt.values)
       #print(tt['Commercial'])           
       #print(tt['Private']) 
       #print(xt1)
       
       split2_p1=tt2['Commercial']/tt2.sum()
       split2_p2=tt2['Private']/tt2.sum()
       split1_p1=tt1['Commercial']/tt1.sum()
       split1_p2=tt1['Private']/tt1.sum()           
       #print(split1_p1)
       #print(split1_p2)
       split1_e=calEntropy([split1_p1,split1_p2])
       split2_e=calEntropy([split2_p1,split2_p2])
       #print(split1_e)
       #print(split2_e)
       #print(len(t2.index))
       #print(len(t3.index))
       norm=len(t2.index)+len(t3.index)
       #print("Norm Fact",norm)
       #print(len(result_XY.index))
       
       split_entropy=((len(t2.index)/norm)*split1_e)+((len(t3.index)/norm)*split2_e)
       #print(len(t2.index))
       #print(len(t3.index))
       #print(len(result_XY.index))
       #print(split_entropy)
       #print("Split Entropy:",split_entropy)
       reduced_entropy=parent_entropy1-split_entropy
       #print(reduced_entropy)
       if reduced_entropy>minimum:
           minimum=reduced_entropy
           best_split[0]=split1
           best_split[1]=split2
           
       print("Reduced Entropy",minimum)
       print("Best Split",best_split)
       print(best_split)
print("Final Reduced Entropy",minimum)
print("Final Best Split",best_split)         
"""                      


def selectNominSplit(result_XY,col_name,parent_entropy):
    cart=result_XY[col_name].unique()
#    cart=predictor_column.unique()
    cart_temp=cart.tolist()
    best_split=[0,0]
    minimum=0
    for i in range(int((len(cart)-1))):
        for j in combinations(cart,i+1): 
            split1=list(j)
            split2=list(set(cart_temp)-set(split1))
        
            t2=result_XY[[col_name,'CAR_USE']][result_XY[col_name].isin(split1)]
            t3=result_XY[[col_name,'CAR_USE']][result_XY[col_name].isin(split2)]
            #print(t2)
            xt1=t2['CAR_USE'].value_counts()
            xt2=t3['CAR_USE'].value_counts()
            xt1.sort_index(inplace=True)
            xt2.sort_index(inplace=True)
            tt1=pandas.Series([0,0],index=['Commercial', 'Private'])
            tt2=pandas.Series([0,0],index=['Commercial', 'Private'])
            for i in range(len(xt1.index)):
#            for i in xt1.index:
                tt1[i]=xt1[i]
           
            for j in range(len(xt2.index)):
#            for i in xt2.index:
                tt2[j]=xt2[j]
                
            split2_p1=tt2['Commercial']/tt2.sum()
            split2_p2=tt2['Private']/tt2.sum()
            split1_p1=tt1['Commercial']/tt1.sum()
            split1_p2=tt1['Private']/tt1.sum()  

            split1_e=calEntropy([split1_p1,split1_p2])
            split2_e=calEntropy([split2_p1,split2_p2])
            norm=len(t2.index)+len(t3.index)
            #split_entropy=(len(t2.index)/len(result_XY.index))*split1_e+(len(t3.index)/len(result_XY.index))*split2_e
            split_entropy=((len(t2.index)/norm)*split1_e)+((len(t3.index)/norm)*split2_e)
            reduced_entropy=parent_entropy-split_entropy
            if reduced_entropy>minimum:
                minimum=reduced_entropy
                best_split[0]=split1
                best_split[1]=split2
            #print("REntropy",minimum)
            #print(best_split)
            #print(reduced_entropy)
                
    return minimum, best_split


def sortOrder(in_list):
    new=['Below High School','High School', 'Bachelors', 'Masters', 'Doctors']
    cart_temp=[]
    for j in new:
        for i in in_list:
            if j==i:
                cart_temp.append(j)
    return cart_temp


def selectOrdinSplit(result_XY,col_name,parent_entropy):
    cart=result_XY[col_name].unique()
#    cart=predictor_column.unique()
    new=['Below High School','High School', 'Bachelors', 'Masters', 'Doctors']
    cart_temp=sortOrder(cart)
  
    best_split=[0,0]
    minimum=0
    for i in range(len(cart)-1):
#        for j in combinations(cart,i+1): 
            split1=cart_temp[:i+1]
            split2=list(set(cart_temp)-set(split1))
            split2=sortOrder(split2)
       
            t2=result_XY[[col_name,'CAR_USE']][result_XY[col_name].isin(split1)]
            t3=result_XY[[col_name,'CAR_USE']][result_XY[col_name].isin(split2)]
            xt1=t2['CAR_USE'].value_counts()
            xt2=t3['CAR_USE'].value_counts()
            xt1.sort_index(inplace=True)
            xt2.sort_index(inplace=True)
            tt1=pandas.Series([0,0],index=['Commercial', 'Private'])
            tt2=pandas.Series([0,0],index=['Commercial', 'Private'])
            for i in range(len(xt1.index)):
                tt1[i]=xt1[i]
           
            for j in range(len(xt2.index)):
                tt2[j]=xt2[j]
                
            split2_p1=tt2['Commercial']/tt2.sum()
            split2_p2=tt2['Private']/tt2.sum()
            split1_p1=tt1['Commercial']/tt1.sum()
            split1_p2=tt1['Private']/tt1.sum()  

            split1_e=calEntropy([split1_p1,split1_p2])
            split2_e=calEntropy([split2_p1,split2_p2])
            norm=len(t2.index)+len(t3.index)
            split_entropy=((len(t2.index)/norm)*split1_e)+((len(t3.index)/norm)*split2_e)
            #split_entropy=(len(t2.index)/len(result_XY.index))*split1_e+(len(t3.index)/len(result_XY.index))*split2_e
            reduced_entropy=parent_entropy-split_entropy
            if reduced_entropy>minimum:
                minimum=reduced_entropy
                best_split[0]=split1
                best_split[1]=split2
            #print("Reduced Entropy",minimum)
            #print(best_split)
            #print(reduced_entropy)
                
    return minimum, best_split


def predict_class(predData):
    if predData['OCCUPATION'] in ('Blue Collar', 'Student', 'Unknown'):
        if predData['EDUCATION'] in ('Below High School'):
            return [0.2693548387096774, 0.7306451612903225]
        else:
            return [0.8376594808622966, 0.16234051913770348]
    else:
        if predData['CAR_TYPE'] in ('Minivan', 'SUV', 'Sports Car'):
            return [0.0845771144278607, 0.9154228855721394]
        else:
            return [0.6100917431192661, 0.38990825688073394]

def predict_class_decision_tree(predData):
    out_data = numpy.ndarray(shape=(len(predData), 2), dtype=float)
    counter = 0
    for index, row in predData.iterrows():
        probability = predict_class(predData=row)
        out_data[counter] = probability
        counter += 1
    return out_data


result_XY = pandas.concat([X_train, Y_train], axis=1, sort=False)
print(result_XY['CAR_USE'].value_counts())
priv_p=result_XY['CAR_USE'].value_counts()[0]/result_XY['CAR_USE'].value_counts().sum()
com_p=result_XY['CAR_USE'].value_counts()[1]/result_XY['CAR_USE'].value_counts().sum()
print("----------------------------Part a----------------------------\n")
print("Entropy of Root Node:")
parent_entropy=calEntropy([priv_p,com_p])
print(parent_entropy)

GenerateTree(result_XY)
#print(node_split)

print("--------------------------Part F------------------------------------")
Y_test=Y_test.to_numpy()
predProb_test = predict_class_decision_tree(predData=X_test)
predProb_y = predProb_test[:, 0] 

# Generate the coordinates for the ROC curve
fpr, tpr, thresholds = metrics.roc_curve(Y_test, predProb_y, pos_label = 'Commercial')             
               
# Draw the Kolmogorov Smirnov curve
cutoff = numpy.where(thresholds > 1.0, numpy.nan, thresholds)
plt.plot(cutoff, tpr, marker = 'o', label = 'True Positive',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot(cutoff, fpr, marker = 'o', label = 'False Positive',
         color = 'red', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.grid(True)
plt.xlabel("Probability Threshold")
plt.ylabel("Positive Rate")
plt.legend(loc = 'upper right', shadow = True, fontsize = 'large')
plt.show()  

#From the above plot we get KS Cut Off around 0.27
threshold1=0.27

print("Kolmogorov-Smirnov statistic from the below plot is around (max difference bwteen TP and FP) : 0.7 ")


print("---------------------------------Question3-----------------------------------------")

print("----------------------------------Part A-------------------------------------------")
#Taking Target Event Varibale as Commercial 
#Event probabilty is calculated as 0.367849 for Commercial in Question1 PartA
threshold=0.367849
predProb_test = predict_class_decision_tree(predData=X_test)
predProb_y = predProb_test[:, 0] 


# determining the predicted class
pred_y = numpy.empty_like(Y_test)
#pred_y = numpy.empty(Y_test.shape,dtype=str)
for i in range(Y_test.shape[0]):
    if predProb_y[i] > threshold:
        pred_y[i] = 'Commercial'
    else:
        pred_y[i] = 'Private'
        
# Calculating accuracy and misclassification rate
accuracy = metrics.accuracy_score(Y_test, pred_y)
misclassification_rate = 1 - accuracy
print(f'Accuracy: {accuracy}')
print(f'Misclassification Rate: {misclassification_rate}')

print("------------------------------------Part B-------------------------------")

#From the above plot we get KS Cut Off around 0.27
threshold1=0.27

# Determining the predicted class
pred_y = numpy.empty_like(Y_test)
#pred_y = numpy.empty(Y_test.shape,dtype=str)
for i in range(Y_test.shape[0]):
    if predProb_y[i] > threshold1:
        pred_y[i] = 'Commercial'
    else:
        pred_y[i] = 'Private'

# Calculating accuracy and misclassification rate
accuracy = metrics.accuracy_score(Y_test, pred_y)
misclassification_rate = 1 - accuracy
print(f'Accuracy: {accuracy}')
print(f'Misclassification Rate: {misclassification_rate}')
           
"""               
# Generate the coordinates for the ROC curve
fpr, tpr, thresholds = metrics.roc_curve(Y_test, predProb_y, pos_label = 'Commercial')             
               
# Draw the Kolmogorov Smirnov curve
cutoff = numpy.where(thresholds > 1.0, numpy.nan, thresholds)
plt.plot(cutoff, tpr, marker = 'o', label = 'True Positive',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot(cutoff, fpr, marker = 'o', label = 'False Positive',
         color = 'red', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.grid(True)
plt.xlabel("Probability Threshold")
plt.ylabel("Positive Rate")
plt.legend(loc = 'upper right', shadow = True, fontsize = 'large')
plt.show()              
"""          

print("---------------------------Part C-----------------------------------")

# Calculating the root average squared error
RASE = 0.0

for y, ppy in zip(Y_test, predProb_y):
    if y == 'Commercial':
        RASE += (1 - ppy) ** 2
    else:
        RASE += (0 - ppy) ** 2
RASE = numpy.sqrt(RASE / Y_test.shape[0])
print(f'Root Average Squared Error: {RASE}')                         


print("-----------------------------Part D-------------------------------------")
y_true = 1.0 * numpy.isin(Y_test, ['Commercial'])
AUC = metrics.roc_auc_score(y_true, predProb_y)
print(f'Area Under Curve: {AUC}')
   
print("-----------------------------Part E-------------------------------------")
    
target_and_predictedprob=numpy.concatenate((Y_test, predProb_y.reshape(Y_test.shape)), axis=1)

com=target_and_predictedprob[target_and_predictedprob[:,0] == 'Commercial']
com[:,1]=numpy.sort(com[:,1])
priv=target_and_predictedprob[target_and_predictedprob[:,0] == 'Private']
priv[:,1]=numpy.sort(priv[:,1])


con=0
dis=0
tie=0

for i in com[:,1]:
    for j in priv[:,1]:
        if i>j:
            con+=1
        elif i==j:
            tie+=1
        else:
            dis+=1

            
print("Gini Coeffiecient:")  
pairs=con+dis+tie
print((con-dis)/pairs)

print("---------------------------Part F-----------------------")
print("Goodman-Kruskal Gamma statistic :")
print((con-dis)/(con+dis))


print("------------------------Part E----------------------------")
  
# Generate the coordinates for the ROC curve
one_minus_specificity, sensitivity, thresholds = metrics.roc_curve(Y_test, predProb_y, pos_label='Commercial')

# Add two dummy coordinates
one_minus_specificity = numpy.append([0], one_minus_specificity)
sensitivity = numpy.append([0], sensitivity)

one_minus_specificity = numpy.append(one_minus_specificity, [1])
sensitivity = numpy.append(sensitivity, [1])

# Draw the ROC curve
plt.figure(figsize=(6, 6))
plt.plot(one_minus_specificity, sensitivity, marker='o', color='blue', linestyle='solid', linewidth=2, markersize=6)
plt.plot([0, 1], [0, 1], color='red', linestyle=':')
plt.grid(True)
plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
ax = plt.gca()
ax.set_aspect('equal')
plt.show()    