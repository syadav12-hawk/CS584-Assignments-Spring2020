# -*- coding: utf-8 -*-
"""
Name : Sourav Yadav 
ID : A20450418
CS584-04  Spring 2020
Assignment 1

"""
import pandas 
import numpy as np
import scipy.stats 


#Function to replace with its Frequency Value
def freq_replace(ser):
    x=ser.value_counts()
    for i in x.index.values:
        ser=ser.replace(i,x.get(i))
    return ser

#Function to return Frequency of a catgory
def name_and_freq(name_ser,freq_ser,col):
    t=name_ser
    t1=freq_ser
#    t2=t1.values.tolist()
    t["Freq"]=t1
    return t
#Function to get frequency of a values
def get_freq(cars,cars_code_freq,col1,name):
    t9=name_and_freq(cars[[col1]],cars_code_freq[[col1]], col1)
    t10=t9.values.tolist()
    for i in t10:
        if i[0] == name:
            return i[1]

#Function for Distance with two "points"
def dist_between(k_1,k_2):
    d=0
    for i in range(len(k_1)):
        d=d+dist_bet_two_freq(k_1[i],k_2[i])
    return d

#Function for distance between two Values
def dist_bet_two_freq(freq1,freq2):
    if freq1==freq2:
        return 0
    else:
        return (1/freq1)+(1/freq2)


#Function to calculate mode between two points
def new_mode_cal(ar):
    mode=np.array([0,0,0,0])
    for i in range(4):
        c,_=scipy.stats.mode(ar[:,i])
        mode[i]=c
        
    return mode

#Function to return Name from Freq
def get_name_from_freq(cars,cars_code_freq,col1,fre):
    t5=name_and_freq(cars[[col1]],cars_code_freq[[col1]], col1)
    t6=t5.values.tolist()
    for i in range(428):
        if (t6[i][1] == fre):
            return t6[i][0]   

#K Mode Function
def k_mode_clustring(freq_array):   
    #Intialize centrois to random values
    k1=freq_array[0]
    k2=freq_array[1]
    k3=freq_array[2] 
    #My estimate is that after 100 iteration centriod converges and hence I have taken 100 loops
    for i in range(100):
        #List to store cluster points
        k1_clus=[]
        k2_clus=[]
        k3_clus=[]
        #Assigning each points to one of three clusters
        for j in range(freq_array.shape[0]):
            #Taking each vector as m input
            m=freq_array[j]
            #Calulating distances from all three centroids
            t1=dist_between(m,k1)
            t2=dist_between(m,k2)
            t3=dist_between(m,k3)
            #Chosing min of three distances
            t4=min(t1,t2,t3)
            #Assigning points to one of three clusters
            if t4==t1:
                k1_clus.append(m)
            elif t4==t2:
                k2_clus.append(m)
            else:
                k3_clus.append(m)  
        #Updating new centroids by calculating modes
        if len(k1_clus)==0:
            k1_new=k1
        else:
            k1_new=new_mode_cal(np.asarray(k1_clus))
            
        if len(k2_clus)==0:
            k2_new=k2
        else:
            k2_new=new_mode_cal(np.asarray(k2_clus))
        
        if len(k3_clus)==0:
            k3_new=k3
        else:
            k3_new=new_mode_cal(np.asarray(k3_clus))
           
        k1=k1_new
        k2=k2_new
        k3=k3_new
        
    #Returning Centroids of clusters as k1,k2,k3 and cluster points as k1_clus,k2_clus,k3_clus    
    return k1,k2,k3,k1_clus,k2_clus,k3_clus


#Import data from CSV
cars_1 = pandas.read_csv('C:\\Users\\soura\\OneDrive\\Desktop\\ML\\ML Spring20_Assignemnt\\HW2\\cars.csv',
                       delimiter=',')

#Taking only the two columns :'Type', 'Origin','DriveTrain','Cylinders'
cars=cars_1[['Type', 'Origin','DriveTrain','Cylinders']] 

#Replacing NaN values with zeros
cars['Cylinders'].fillna(0, inplace = True)

cars_code_freq=cars.copy()

#Get columns names
col_name=list(cars.columns.values)

#Replace values with Frequencies
for i in col_name:
    cars_code_freq[i]= freq_replace(cars_code_freq[i])
    
#-------------------------------PartA-----------------------------------------

t=name_and_freq(cars[['Type']],cars_code_freq[['Type']],'Type')
print(t.Freq.unique())

print(f"Frequencies of the categorical feature 'Type': {t[['Type','Freq']]}")

#-----------------------------PartB------------------------------------------

t3=name_and_freq(cars[['DriveTrain']],cars_code_freq[['DriveTrain']],'DriveTrain')
print(t3.Freq.unique())
print(f"Frequencies of the categorical feature 'Type': {t3[['DriveTrain','Freq']]}")

#------------------------------------------PartC---------------------------------

#asia_fre=name_and_freq(cars[['Origin']],cars_code_freq[['Origin']], 'Origin')
#Get Frequency of Asia and Europe
freq_asia=get_freq(cars, cars_code_freq, 'Origin', 'Asia')
freq_europe=get_freq(cars, cars_code_freq,'Origin', 'Europe')

#Disatnce between Asia and Europe
dis_ae=dist_bet_two_freq(freq_asia, freq_europe)
print(f"Distance between Asia and Europe is : {dis_ae}")

#---------------------------------PartD-----------------------------------
#Get frequencies of Cylinders5 and Cylinders0
freq_cyc5=get_freq(cars, cars_code_freq, 'Cylinders', 5)
freq_cyc0=get_freq(cars, cars_code_freq,'Cylinders', 0)

#Distance between Cylinder5 and Cylinder0
print(f"Distance between Cylinder5 and Cylinder0 is : {dist_bet_two_freq(freq_cyc5, freq_cyc0)}")

#-----------------------------------PartE-----------------------------------

freq_array=cars_code_freq.to_numpy()

#Calculate Number of Observation in each cluster

#freq_array=preprocessing.normalize(freq_array)
cl1,cl2,cl3,cl1_arr,cl2_arr,cl3_arr=k_mode_clustring(freq_array)

print(f"Number of Observations in Cluster1: {len(cl1_arr)}")
print(f"Number of Observations in Cluster2: {len(cl2_arr)}")
print(f"Number of Observations in Cluster3: {len(cl3_arr)}")



    
#Priting Centroid Frequency
c1=[]
for i in cl1:
    c1.append(float(i))
print(f"Centroid of CLuster 1: {c1}")

#Priting Centroid Names
count1=0
cent_col=[]
for i in col_name:   
    count2=0
    for j in c1:
        if count1==count2:
            cent_col.append(get_name_from_freq(cars,cars_code_freq,i,j))
            
        count2+=1 
    count1+=1    
print(cent_col)   

#Priting Centroid Frequency
c2=[]
for i in cl2:
    c2.append(float(i))
print(f"Centroid of CLuster 2: {c2}")

#Priting Centroid Names
count1=0
cent_col1=[]
for i in col_name:   
    count2=0
    for j in c2:
        if count1==count2:
            cent_col1.append(get_name_from_freq(cars,cars_code_freq,i,j))
            
        count2+=1 
    count1+=1    
print(cent_col1)  

#Priting Centroid Frequency
c3=[]
for i in cl3:
    c3.append(float(i))
print(f"Centroid of CLuster 3: {c3}")


#Priting Centroid Names
count1=0
cent_col2=[]
for i in col_name:   
    count2=0
    for j in c3:
        if count1==count2:
            cent_col2.append(get_name_from_freq(cars,cars_code_freq,i,j))
            
        count2+=1 
    count1+=1    
print(cent_col2)  

#-----------------------------------PartF-----------------------------------

#Displaying Freq Distributation table of Origin features in each cluster
cl1_points=[]
c1_a=0
c1_e=0
c1_u=0
for i in range(len(cl1_arr)):
    cl1_points.append(cl1_arr[i][1])
    if cl1_arr[i][1]==158:
        c1_a+=1
    elif cl1_arr[i][1]==123:
        c1_e+=1
    else:
        c1_u+=1
        
print(f"Frequency Distribution  of Cluster 1: Asia :{c1_a}, Europe:{c1_e}, USA:{c1_u}")
        
    
cl2_points=[]
c2_a=0
c2_e=0
c2_u=0
for i in range(len(cl2_arr)):
    cl2_points.append(cl2_arr[i][1])
    if cl2_arr[i][1]==158:
        c2_a+=1
    elif cl2_arr[i][1]==123:
        c2_e+=1
    else:
        c2_u+=1

print(f"Frequency Distribution  of Cluster 2: Asia :{c2_a}, Europe:{c2_e}, USA:{c2_u}")
    
cl3_points=[]
c3_a=0
c3_e=0
c3_u=0
for i in range(len(cl3_arr)):
    cl3_points.append(cl3_arr[i][1])
    if cl3_arr[i][1]==158:
        c3_a+=1
    elif cl3_arr[i][1]==123:
        c3_e+=1
    else:
        c3_u+=1

print(f"Frequency Distribution  of Cluster 3: Asia :{c3_a}, Europe:{c3_e}, USA:{c3_u}")

#Cluster 1 Frequency Distribuation   
import matplotlib.pyplot as plt  
plt.hist(cl1_points,edgecolor='k')
plt.ylabel("Number of Occurance of Given Frequency")
plt.xlabel("Frequency of Origin")
plt.title("Cluster 1")
plt.show() 
plt.clf() 



#Cluster 2 Frequency Distribuation   
import matplotlib.pyplot as plt  
plt.hist(cl2_points,edgecolor='k')
plt.ylabel("Number of Occurance of Given Frequency")
plt.xlabel("Frequency of Origin")
plt.title("Cluster 2")
plt.show() 
plt.clf() 

#Cluster 3 Frequency Distribuation   
import matplotlib.pyplot as plt  
plt.hist(cl3_points,edgecolor='k')
plt.ylabel("Number of Occurance of Given Frequency")
plt.xlabel("Frequency of Origin")
plt.title("Cluster 3")
plt.show() 
plt.clf() 