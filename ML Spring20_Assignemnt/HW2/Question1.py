# -*- coding: utf-8 -*-
"""
Name : Sourav Yadav 
ID : A20450418
CS584-04 Spring 2020
Assignment 2

"""

#--------------------------------Part A-------------------------------------

# Import Data 
import pandas 
import numpy as np

grocer_cust_item = pandas.read_csv('C:\\Users\\soura\\OneDrive\\Desktop\\ML\\ML Spring20_Assignemnt\\HW2\\Groceries.csv',
                       delimiter=',')

# Convert the Sale Receipt data to the Item List format
groc_ListItem = grocer_cust_item.groupby(['Customer'])['Item'].apply(list).values.tolist()

# Convert the Item List format to the Item Indicator format
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(groc_ListItem).transform(groc_ListItem)
groc_ItemIndicator = pandas.DataFrame(te_ary, columns=te.columns_)


cost_items=[]
cost_item_count=[]
for item in groc_ListItem:
    uniq_items=set(item)
    cost_item_count.append(len(uniq_items))
    cost_items.append(list(uniq_items))
  
uniq_item_df=pandas.DataFrame(cost_items)

import matplotlib.pyplot as plt  
plt.hist(cost_item_count,edgecolor='k')
plt.xlabel("Unique Items")
plt.ylabel("Customer")
plt.show()  


_25_per,_50_per,_75_per=np.percentile(cost_item_count,(25,50,75)) 

print(f"25th Percentaile:{_25_per}\n50th Percentiles:{_50_per}\n75th Percentiles:{_75_per}")

#----------------Part B------------------------------------------

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

number_of_cust=groc_ItemIndicator.shape[0]
# Find the frequent itemsets
frequent_itemsets = apriori(groc_ItemIndicator, min_support = (75/number_of_cust), use_colnames = True)
print(f"Number of itemsets found:{frequent_itemsets.shape[0]}")

# Finding largest K value
max=0
s=frequent_itemsets['itemsets'].to_numpy()
f_list=[list(x) for x in s]
for j in f_list:
    if len(j)>max:
        max=len(j)
print(f"The largest k value among our itemsets:{max}") 

#----------------------------PartC-------------------------

# Discover the association rules
assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.01)
print(f"Number of Association Rules: {assoc_rules.shape[0]}")

#--------------------------Part D----------------------------

#Plot the Support metrics Vs the Confidence metrics 
import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
x=plt.scatter(assoc_rules['confidence'], assoc_rules['support'], c = assoc_rules['lift'])
cb=plt.colorbar(x)
cb.set_label('Lift')
plt.grid(True)
plt.xlabel("Confidence")
plt.ylabel("Support")
plt.show()

#-------------------------Part E------------------------------

#List of the rules whose Confidence metrics are greater than or equal to 60%.
row_index=[]
c=0
for value in assoc_rules[['confidence']].values:
    if value>=0.6:
        row_index.append(c)
    c+=1
        

for j in row_index :
    print(assoc_rules.loc[j,['antecedents','consequents','support','lift']])
    print("\n") 
     








