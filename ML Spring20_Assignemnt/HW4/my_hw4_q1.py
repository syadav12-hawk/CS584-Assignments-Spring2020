
"""
Name : Sourav Yadav
AID : A20450418
Homework 4
Spring 2020

"""

#Importing Packages
import numpy
import pandas
import scipy
import sympy 
import statsmodels.api as stats
import math


# A function that returns the columnwise product of two dataframes (must have same number of rows)
def create_interaction (inDF1, inDF2):
    name1 = inDF1.columns
    name2 = inDF2.columns
    outDF = pandas.DataFrame()
    for col1 in name1:
        for col2 in name2:
            outName = col1 + " * " + col2
            outDF[outName] = inDF1[col1] * inDF2[col2]
    return(outDF)

# A function that find the non-aliased columns, fit a logistic model, and return the full parameter estimates
def build_mnlogit (fullX, y, debug = 'N'):
    # Number of all parameters
    nFullParam = fullX.shape[1]

    # Number of target categories
    y_category = y.cat.categories
    nYCat = len(y_category)

    # Find the non-redundant columns in the design matrix fullX
    reduced_form, inds = sympy.Matrix(fullX.values).rref()

    # These are the column numbers of the non-redundant columns
    if (debug == 'Y'):
        print('Column Numbers of the Non-redundant Columns:')
        print(inds)

    # Extract only the non-redundant columns for modeling
    X = fullX.iloc[:, list(inds)]

    # The number of free parameters
    thisDF = len(inds) * (nYCat - 1)

    # Build a multionomial logistic model
    logit = stats.MNLogit(y, X)
    thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
    thisParameter = thisFit.params
    thisLLK = logit.loglike(thisParameter.values)

    if (debug == 'Y'):
        print(thisFit.summary())
        print("Model Parameter Estimates:\n", thisParameter)
        print("Model Log-Likelihood Value =", thisLLK)
        print("Number of Free Parameters =", thisDF)

    # Recreat the estimates of the full parameters
    workParams = pandas.DataFrame(numpy.zeros(shape = (nFullParam, (nYCat - 1))))
    workParams = workParams.set_index(keys = fullX.columns)
    fullParams = pandas.merge(workParams, thisParameter, how = "left", left_index = True, right_index = True)
    fullParams = fullParams.drop(columns = '0_x').fillna(0.0)

    # Return model statistics
    return (thisLLK, thisDF, fullParams)

#Load Data
dataframe = pandas.read_csv('C:/Users/soura/OneDrive/Desktop/ML/ML Spring20_Assignemnt/HW4/Purchase_Likelihood.csv')
dataframe = dataframe.dropna()

# Specify Origin as a categorical variable
y = dataframe['insurance'].astype('category')

# Specify Group_size, Homeowner and married_couple as categorical variables
d_group_size = pandas.get_dummies(dataframe[['group_size']].astype('category'))
d_homeowner = pandas.get_dummies(dataframe[['homeowner']].astype('category'))
d_married_couple = pandas.get_dummies(dataframe[['married_couple']].astype('category'))

# Intercept only model
designX = pandas.DataFrame(y.where(y.isnull(), 1))
LLK0, DF0, fullParams0 = build_mnlogit (designX, y, debug = 'Y')
#(LLK0,DF0)
print(LLK0)
print(DF0)
print(fullParams0)

#Adding Deviance test statistic in DataFrame
tdf = pandas.Series([DF0,LLK0,0,0,0],index=["Free Parameter","Log-Likelihood","Deviance","Degrees of Freedom","Significance"])
df = pandas.DataFrame([tdf],index = ["Intercept"])

#Calculating the Feature Importance Index 
tdf1 = pandas.Series([0],index=["Importance"])
df1 = pandas.DataFrame([tdf1],index = ["Intercept"])

#--------------------------------------------------------------------------------------
# Intercept + Group_size
designX = stats.add_constant(d_group_size, prepend=True)
LLK_1R, DF_1R, fullParams_1R = build_mnlogit (designX, y, debug = 'Y')
testDev = 2 * (LLK_1R - LLK0)
testDF = DF_1R - DF0
testPValue = scipy.stats.chi2.sf(testDev, testDF)
aliased_col = fullParams_1R[(fullParams_1R['0_y'] == 0)&(fullParams_1R['1_y'] == 0)].index.tolist()

print('Model: Intercept + Group_size ')
print('Deviance Chi=Square Test')
print('Number of Free Parameters :',DF_1R)
print('Log Liklihood :',LLK_1R)
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
print('List of Aliased Columns for the model',aliased_col)



tdf = pandas.Series([DF_1R,LLK_1R,testDev,testDF,testPValue],index=["Free Parameter","Log-Likelihood","Deviance","Degrees of Freedom","Significance"])
row_df = pandas.DataFrame([tdf],index = ["Group_size"])
df = pandas.concat([df,row_df])

#Calculating second table
tdf1 = pandas.Series([-math.log10(testPValue)],index=["Importance"])
row_df1 = pandas.DataFrame([tdf1],index = ["Group_size"])
df1 = pandas.concat([df1,row_df1])

#------------------------------------------------------------------------------------
# Intercept + Group_size + homeowner
designX = d_group_size
designX = designX.join(d_homeowner)
designX = stats.add_constant(designX, prepend=True)
LLK_1R_1J, DF_1R_1J, fullParams_1R_1J = build_mnlogit (designX, y, debug = 'Y')
testDev = 2 * (LLK_1R_1J - LLK_1R)
testDF = DF_1R_1J - DF_1R
testPValue = scipy.stats.chi2.sf(testDev, testDF)
aliased_col = fullParams_1R_1J[(fullParams_1R_1J['0_y'] == 0)&(fullParams_1R_1J['1_y'] == 0)].index.tolist()

print('Model : Intercept + Group_size + homeowner ')
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
print('List of Aliased Columns for the model : Intercept + Group_size + homeowner ')
print('List of Aliased Columns for the model',aliased_col)

tdf = pandas.Series([DF_1R_1J,LLK_1R_1J,testDev,testDF,testPValue],index=["Free Parameter","Log-Likelihood","Deviance","Degrees of Freedom","Significance"])
row_df = pandas.DataFrame([tdf],index = ["Homeowner"])
df = pandas.concat([df,row_df])

#Calculating second table
try :    
    tdf1 = pandas.Series([-math.log10(testPValue)],index=["Importance"])
except :
    tdf1 = pandas.Series([0],index=["Importance"])
row_df1 = pandas.DataFrame([tdf1],index = ["Homeowner"])
df1 = pandas.concat([df1,row_df1])


#----------------------------------------------------------------------------------------
# Intercept + Group_size + homeowner + married_couple
designX = d_group_size
designX = designX.join(d_homeowner)
designX = designX.join(d_married_couple)
designX = stats.add_constant(designX, prepend=True)

LLK_1R_1J_1M, DF_1R_1J_1M, fullParams_1R_1J_1M = build_mnlogit (designX, y, debug = 'Y')
testDev = 2 * (LLK_1R_1J_1M - LLK_1R_1J)
testDF = DF_1R_1J_1M - DF_1R_1J
testPValue = scipy.stats.chi2.sf(testDev, testDF)
aliased_col = fullParams_1R_1J_1M[(fullParams_1R_1J_1M['0_y'] == 0)&(fullParams_1R_1J_1M['1_y'] == 0)].index.tolist()

print('Model : Intercept + Group_size + homeowner + married_couple ')
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
print('List of Aliased Columns for the model',aliased_col)

tdf = pandas.Series([DF_1R_1J_1M,LLK_1R_1J_1M,testDev,testDF,testPValue],index=["Free Parameter","Log-Likelihood","Deviance","Degrees of Freedom","Significance"])
row_df = pandas.DataFrame([tdf],index = ["Married_couple"])
df = pandas.concat([df,row_df])


#Calculating second table
tdf1 = pandas.Series([-math.log10(testPValue)],index=["Importance"])
row_df1 = pandas.DataFrame([tdf1],index = ["Married Couple"])
df1 = pandas.concat([df1,row_df1])


#------------------------------------------------------------------------------------
# Intercept + Group_size + homeowner + married_couple + Group_size*homeowner
designX = d_group_size
designX = designX.join(d_homeowner)
designX = designX.join(d_married_couple)

d_gs_ho = create_interaction(d_group_size, d_homeowner)
designX = designX.join(d_gs_ho)
designX = stats.add_constant(designX, prepend=True)

LLK_2RJ, DF_2RJ, fullParams_2RJ = build_mnlogit (designX, y, debug = 'Y')
testDev = 2 * (LLK_2RJ - LLK_1R_1J_1M)
testDF = DF_2RJ - DF_1R_1J_1M
testPValue = scipy.stats.chi2.sf(testDev, testDF)
aliased_col = fullParams_2RJ[(fullParams_2RJ['0_y'] == 0)&(fullParams_2RJ['1_y'] == 0)].index.tolist()

print('Model: Intercept + Group_size + homeowner + married_couple + Group_size*homeowner ')
print('Deviance Chi=Square Test') 
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
print('List of Aliased Columns for the model',aliased_col)

tdf = pandas.Series([DF_2RJ,LLK_2RJ,testDev,testDF,testPValue],index=["Free Parameter","Log-Likelihood","Deviance","Degrees of Freedom","Significance"])
row_df = pandas.DataFrame([tdf],index = ["Group_size*Homeowner"])
df = pandas.concat([df,row_df])

#Calculating second table
tdf1 = pandas.Series([-math.log10(testPValue)],index=["Importance"])
row_df1 = pandas.DataFrame([tdf1],index = ["Group_size*Homeowner"])
df1 = pandas.concat([df1,row_df1])


#-------------------------------------------------------------------------------------------
# Intercept + Group_size + homeowner + married_couple + Group_size*homeowner + Group_size*married_couple
designX = d_group_size
designX = designX.join(d_homeowner)
designX = designX.join(d_married_couple)

d_gs_ho = create_interaction(d_group_size, d_homeowner)
designX = designX.join(d_gs_ho)
#designX = stats.add_constant(designX, prepend=True)

d_gs_mc = create_interaction(d_group_size, d_married_couple)
designX = designX.join(d_gs_mc)
designX = stats.add_constant(designX, prepend=True)

LLK_2RJB, DF_2RJB, fullParams_2RJM = build_mnlogit (designX, y, debug = 'Y')
testDev = 2 * (LLK_2RJB - LLK_2RJ)
testDF = DF_2RJB - DF_2RJ
testPValue = scipy.stats.chi2.sf(testDev, testDF)
aliased_col = fullParams_2RJM[(fullParams_2RJM['0_y'] == 0)&(fullParams_2RJM['1_y'] == 0)].index.tolist()

print('Model : Intercept + Group_size + homeowner + married_couple + Group_size*homeowner + Group_size*married_couple')
print('Deviance Chi=Square Test') 
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
print('List of Aliased Columns for the model :',aliased_col)

tdf = pandas.Series([DF_2RJB,LLK_2RJB,testDev,testDF,testPValue],index=["Free Parameter","Log-Likelihood","Deviance","Degrees of Freedom","Significance"])
row_df = pandas.DataFrame([tdf],index = ["Group_size*Married_couple"])
df = pandas.concat([df,row_df])

#Calculating second table
tdf1 = pandas.Series([-math.log10(testPValue)],index=["Importance"])
row_df1 = pandas.DataFrame([tdf1],index = ["Group_size*Married_couple"])
df1 = pandas.concat([df1,row_df1])


#--------------------------------------------------------------------------------------------------------
# Intercept + Group_size + homeowner + married_couple + Group_size*homeowner + Group_size*married_couple + homeowner*married_couple
designX = d_group_size
designX = designX.join(d_homeowner)
designX = designX.join(d_married_couple)

d_gs_ho = create_interaction(d_group_size, d_homeowner)
designX = designX.join(d_gs_ho)


d_gs_mc = create_interaction(d_group_size, d_married_couple)
designX = designX.join(d_gs_mc)
#designX = stats.add_constant(designX, prepend=True)

d_ho_mc = create_interaction(d_homeowner, d_married_couple)
designX = designX.join(d_ho_mc)
designX = stats.add_constant(designX, prepend=True)

LLK_GGHMC, DF_GGHMC, fullParams_GGHMC = build_mnlogit (designX, y, debug = 'Y')
testDev = 2 * (LLK_GGHMC - LLK_2RJB)
testDF = DF_GGHMC - DF_2RJB
testPValue = scipy.stats.chi2.sf(testDev, testDF)


aliased_col = fullParams_GGHMC[(fullParams_GGHMC['0_y'] == 0)&(fullParams_GGHMC['1_y'] == 0)].index.tolist()

print('Model : Intercept + Group_size + homeowner + married_couple + Group_size*homeowner + Group_size*married_couple + homeowner*married_couple')
print('Deviance Chi=Square Test') 
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
print('List of Aliased Columns for the model :\n')
for i in aliased_col:
    print(i)
    #print("\n")

tdf= pandas.Series([DF_GGHMC,LLK_GGHMC,testDev,testDF,testPValue],index=["Free Parameter","Log-Likelihood","Deviance","Degrees of Freedom","Significance"])
row_df = pandas.DataFrame([tdf],index = ["Homeowner*Married_couple"])
df = pandas.concat([df,row_df])

#Calculating second table
tdf1 = pandas.Series([-math.log10(testPValue)],index=["Importance"])
row_df1 = pandas.DataFrame([tdf1],index = ["Homeowner*Married_couple"])
df1 = pandas.concat([df1,row_df1])
#-------------------------------------------------------------------------
#Exporting to excel
df.to_excel("output_1_c.xlsx")  
print(df)
#Exporting to excel
df1.to_excel("output_1_d.xlsx")  
print(df1)

################################################Question 2#####################################
group_sizes = sorted(list(dataframe.group_size.unique()))
homeowners = sorted(list(dataframe.homeowner.unique()))
married_couples = sorted(list(dataframe.married_couple.unique()))

import itertools
all_combi = list(itertools.product(group_sizes, homeowners, married_couples))

df2 = pandas.DataFrame(all_combi, columns=['group_size','homeowner','married_couple'])

df_groupsize = pandas.get_dummies(df2[['group_size']].astype('category'))
df_homeowner = pandas.get_dummies(df2[['homeowner']].astype('category'))
df_marriedcouple = pandas.get_dummies(df2[['married_couple']].astype('category'))


df_groupsize_homeowner = create_interaction(df_groupsize, df_homeowner)
df_groupsize_marriedcouple = create_interaction(df_groupsize, df_marriedcouple)
df_home_marriedcouple = create_interaction(df_homeowner, df_marriedcouple)

X_Test = df_groupsize.join(df_homeowner)
X_Test = X_Test.join(df_marriedcouple)
X_Test = X_Test.join(df_groupsize_homeowner)

X_Test = X_Test.join(df_groupsize_marriedcouple)
X_Test = X_Test.join(df_home_marriedcouple)
X_Test = stats.add_constant(X_Test, prepend=True)


#Build the final model
logit = stats.MNLogit(y, designX)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)

predictions = thisFit.predict(X_Test)
predictions=pandas.DataFrame.join(pandas.DataFrame(all_combi, columns = ["group_size","homeOwner","Married_couple"]),predictions)

predictions.to_excel("output2_a.xlsx")

#-------------------------------2.b-----------------------------------------------------


predictions['Odds']=predictions.iloc[:,4]/predictions.iloc[:,3]
print('Max Odd Value:',max(predictions['Odds']))
t=predictions['Odds'].idxmax(axis=0)
print('Value of Combination\n',df2.iloc[t,:])

#----------------------------------2.c---------------------------------------------


num=(predictions.loc[predictions['group_size']==3].iloc[:,5])/(predictions.loc[predictions['group_size']==3].iloc[:,3])
den=(predictions.loc[predictions['group_size']==1].iloc[:,5])/(predictions.loc[predictions['group_size']==1].iloc[:,3])
num=num.reset_index()
num=num.iloc[:,1]

odds=pandas.DataFrame(num/den, columns=['Odds_ratio'])

gz_mc_ho=predictions.loc[predictions['group_size']==3].iloc[:,:3]
gz_mc_ho=gz_mc_ho.reset_index()

gz_mc_ho=gz_mc_ho.iloc[:,1:]

result = pandas.concat([gz_mc_ho,odds],axis=1)
print(result.iloc[:,1:])


#-----------------------------------2.d-------------------------------------
num1=(predictions.loc[predictions['homeOwner']==0].iloc[:,3])/(predictions.loc[predictions['homeOwner']==0].iloc[:,4])
den2=(predictions.loc[predictions['homeOwner']==1].iloc[:,3])/(predictions.loc[predictions['homeOwner']==1].iloc[:,4])


num1=num1.reset_index()
num1=num1.iloc[:,1]


den2=den2.reset_index()
den2=den2.iloc[:,1]


odds1=num1/den2

gz_mc_ho1=predictions.loc[predictions['homeOwner']==1].iloc[:,:3]
gz_mc_ho1=gz_mc_ho1.reset_index()

gz_mc_ho1=gz_mc_ho1.iloc[:,1:]

result1 = pandas.concat([gz_mc_ho1,odds1],axis=1)
print(result1)

