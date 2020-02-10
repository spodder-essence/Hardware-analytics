# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 15:21:16 2019

@author: maximilian.schmitt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
import math
import statsmodels.api as sm

def main():

    #########################################################################
    """Add path names, metrics and success values
        Confirm metric names are same across all files
        Change the Creative Test cell column name to *  Version  * across all files
    """

    #Enter path(s) and file(s) for Creative Test Wave(s)
    paths = [
    'C:/Users/showmik.podder/Documents/GitHub/playpen/Max_Regressor/Nest_CA_wave1.csv',
    'C:/Users/showmik.podder/Documents/GitHub/playpen/Max_Regressor/Nest_CA_wave2.csv'
    ]

    #Enter Metrics for Analysis (["Awareness", "Consideration"])
    #metricList = ["Default Weights","Q121","Parents","Q127","Q128","Q129.1","NBA Fans","Q129.3","Q129.4","Q129.5","Q129.6","Q129.7","Q129.8","Version","Creatives","ProductAssociation_SmartSpeakers","ProductAssociation_SmartDisplays","ProductAssociation_Thermostats","ProductAssociation_SecurityCamera","ProductAssociation_Doorbells","ProductAssociation_SecuritySystems","ProductAssociation_Wifi","ProductAssociation_Smartphones","ProductAssociation_Noneoftheabove","ProductAssociation_IDK","Aided Awareness_GoogleNest","Aided Awareness_Amazon","Aided Awareness_Samsung","Aided Awareness_Apple","Aided Awareness_None","Consideration_GoogleNest","Consideration_Amazon","Consideration_Samsung","Consideration_Apple","Consideration_None","Purchase Intent","Perception_Easier","Perception_Helpful","Perception_ProductRange","Perception_Seamless","Perception_NotSeamless","Perception_None","CategoryConsideration_Speakers","CategoryConsideration_Displays","CategoryConsideration_Thermostats","CategoryConsideration_SecurityCameras","CategoryConsideration_Doorbells","CategoryConsideration_SecuritySystem","CategoryConsideration_Wifi","CategoryConsideration_Smartphones","CategoryConsideration_None","Enjoyment","Q166","Q167","Q168","Understanding","Relevance","Branding","Brand Appeal","Distinctiveness","Q175","Q176","Q177"]
    metricList =["Aided Awareness_GoogleNest","Purchase Intent"]
    #Enter Success Values for each metric ([1,2,3,1,0,0,1])
    successList = [2,0]


    #Enter x variable(s) in list format: ["Awareness","Consideration"]
    x = ["Aided Awareness_GoogleNest"]

    #Enter y variable, as a string: "Intent"
    y = "Purchase Intent"



    #Would you like to include control cells in this analysis? Yes = 1, No = 0
    include_control = 0


    #Which of the following test would you like to perform? Yes = 1, No = 0
    Creative_Level_Linear_Regression = 1
    Creative_Level_Linear_Lift_Regression = 1
    Creative_Level_Random_Forest = 0
    Respondent_Level_Linear_Regression = 1
    Respondent_Level_Logistic_Regression = 1
    Respondent_Level_Random_Forest = 0




    ###########################################################################################
    waves = prepCreatives(paths, metricList, successList)

    if(Creative_Level_Linear_Regression):
        creative_linear_regression(waves, x, y, include_control)

    if(Creative_Level_Linear_Lift_Regression):
        creative_linear_regression_lift(waves,x,y,control=False)

    if(Creative_Level_Random_Forest):
        RandomForest(waves, x, y, include_control, user = False)

    if(Respondent_Level_Linear_Regression):
        respondent_linear_regression(waves,x,y,include_control)

    if(Respondent_Level_Logistic_Regression):
        respondent_logistic_regression(waves, x,y,include_control)

    if(Respondent_Level_Random_Forest):
        RandomForest(waves, x, y, include_control, user = True)




class CreativeTest(object):

    def __init__(self, pathName, metricList, successList):

        self.pathName = pathName
        self.df = read(self.pathName)
        self.name = (self.pathName.split('/')[-1]).split(".")[0]
        self.metricList = metricList
        self.successList = successList
        self.versions = (list(set(self.df['Version'].tolist())))
        self.versions.sort()
        self.assetList = []

        """
        for i in range(len(self.versions)):
            #Create assets
            self.assetList.append(Asset(self, i, self.cleanDF(), self.metricList,self.successList))
        """

        #adding full DF for testing purposes
        for i in range(len(self.versions)):
            #Create assets
            self.assetList.append(Asset(self, i, self.df, self.metricList,self.successList))


    def __repr__(self):
        return self.name

    def query(self,metric, id):
        return self.assetList[id].abs_lift[metric]

    def query_lift(self,metric, id):
        return self.assetList[id].lift[metric]

    def cleanDF(self):
        tmp = self.df[['Default Weights','Version']]
        for i in range(len(self.metricList)):
            tmp.insert(i+2,self.metricList[i],self.df[self.metricList[i]])
        return tmp

    def baseline(self,metric):
        return self.assetList[0].abs_lift[metric]




class Asset(object):

    def __init__(self,CreativeTest, id, df, metricList, successList):

        self.CreativeTest = CreativeTest
        self.id = id
        self.df_id = df[df.Version==id]
        self.metricList = metricList
        self.successList = successList
        self.n = len(self.df_id.index)
        self.abs_lift = {}
        self.stat_set = {}
        self.lift = {}

        for i in range(len(metricList)):
            tmp = []
            metric = self.df_id[metricList[i]].tolist()
            success = successList[i]
            weights = self.df_id['Default Weights'].tolist()
            val = 0
            for j in range(self.n):
                if metric[j] == success:
                    val+= weights[j]
                    tmp.append(weights[j])
                else:
                    tmp.append(0)
            self.stat_set[metricList[i]] = tmp
            self.abs_lift[metricList[i]] = (val/self.n)


        df_tmp = pd.DataFrame()
        for i in range(len(metricList)):
            tmp = self.df_id[metricList[i]].tolist()
            for j in range(len(tmp)):
                if (tmp[j] == successList[i]):
                    tmp[j]=1
                else:
                    tmp[j]=0
            df_tmp.insert(i, metricList[i],tmp)
        self.df_id = df_tmp

    def __repr__(self):
        tmp = ""
        tmp = tmp + str(self.CreativeTest.name)
        tmp = tmp + ", Asset "
        tmp = tmp + str(self.id)
        return tmp

    def lift(self):
        for i in range(len(self.metricList)):
            self.lift[self.metricList[i]] = self.CreativeTest.assetList[0]




    def CI(self, metric):
        control = self.CreativeTest.assetList[0]
        p = control.abs_lift[metric]
        n = control.n
        ME = (self.abs_lift[metric])-p
        k = np.sqrt((p*(1-p))/n)
        CV = ME/k


        return (stats.norm.cdf(CV))


def read(filePath):
    res = pd.read_csv(filePath, index_col=False)
    return res



def prepCreatives(array, array2, array3):
    res = []
    for i in range(len(array)):
        res.append(CreativeTest(array[i],array2,array3))
    lift(res)
    return res


def summarize(waves):
    for i in range(len(waves)):
        print(waves[i].name,":")
        for j in range(len(waves[i].metricList)):
            print("\t", waves[i].metricList[j],":")
            for k in range(len(waves[i].assetList)):
                print("\t\tAsset: {0:2d} : {1:8.2f}%".format(waves[i].assetList[k].id, waves[i].assetList[k].abs_lift[waves[i].metricList[j]]*100))

def lift(waves):
    for i in range(len(waves)):
        for j in range(len(waves[i].assetList)):
            for k in range(len(waves[i].assetList[j].metricList)):
                if (waves[i].assetList[j].metricList[k] in waves[i].assetList[j].abs_lift):
                    waves[i].assetList[j].lift[waves[i].assetList[j].metricList[k]] = waves[i].assetList[j].abs_lift[waves[i].assetList[j].metricList[k]] - waves[i].assetList[0].abs_lift[waves[i].assetList[j].metricList[k]]


def baseline(waves, metric):
    n_list = []
    baseline_list = []
    count = 0
    for i in range(len(waves)):
        n_list.append(waves[i].assetList[0].n)
        baseline_list.append(waves[i].baseline(metric))
        count+=n_list[i]
    val = 0.0
    for i in range(len(n_list)):
        val += (n_list[i]/count)*baseline_list[i]
    return val


def query(waves,x, control):
    tmp =[]
    for i in range(len(waves)):
        for j in range(len(waves[i].assetList)):
            if (control == True):
                tmp.append(waves[i].query(x,j))
            else:
                if (waves[i].assetList[j].id!=0):
                    tmp.append(waves[i].query(x,j))
    return tmp


def query_lift(waves,x,control):
    tmp =[]
    for i in range(len(waves)):
        for j in range(len(waves[i].assetList)):
            if (control == True):
                tmp.append(waves[i].query_lift(x,j))
            else:
                if (waves[i].assetList[j].id!=0):
                    tmp.append(waves[i].query_lift(x,j))
    return tmp

def aggregate_creative_data_lift(waves, control=False):
    df = pd.DataFrame()
    count = 0
    for i in range(len(waves[0].metricList)):
        if (waves[0].successList[i] != -1):
            tmp = query_lift(waves, waves[0].metricList[i], control)
            df.insert(count,waves[0].metricList[i],tmp)
            count+=1
    return df

def aggregate_creative_data(waves, control=False):
    df = pd.DataFrame()
    count = 0
    for i in range(len(waves[0].metricList)):
        if (waves[0].successList[i] != -1):
            tmp = query(waves, waves[0].metricList[i], control)
            df.insert(count,waves[0].metricList[i],tmp)
            count+=1
    return df

def creative_correlation(waves,x,y,control):
    df = aggregate_creative_data(waves,control)
    return np.corrcoef(np.asarray(df[y]),np.asarray(df[x]))


def creative_linear_regression_lift(waves,x,y,control):
    print("Creative-level Linear Regression of Lift:")
    df = aggregate_creative_data_lift(waves, control)
    model = sklearn.linear_model.LinearRegression()
    model.fit(df[x],df[y])
    for i in range(len(model.coef_)):
        print(x[i],": ",model.coef_[i])
    print("Intercept: ",model.intercept_)
    print("R2: ",model.score(df[x],df[y]))
    if (len(x)==1):
        b = []
        b.append(model.intercept_)
        b.append(model.coef_)
        plot_regression_line(df[x],df[y],b)


def creative_linear_regression(waves, x, y, control):
    print("Creative-level Linear Regression:")
    df = aggregate_creative_data(waves, control)
    model = sklearn.linear_model.LinearRegression()
    model.fit(df[x],df[y])
    for i in range(len(model.coef_)):
        print(x[i],": ",model.coef_[i])
    print("Intercept: ",model.intercept_)
    print("R2: ",model.score(df[x],df[y]))
    if (len(x)==1):
        b = []
        b.append(model.intercept_)
        b.append(model.coef_)
        plot_regression_line(df[x],df[y],b)
    return print("Success")



def plot_regression_line(x,y,b):
    plt.scatter(x, y, color = "m", marker = "o", s = 30)
    y_pred = b[0] + b[1]*x
    plt.plot(x, y_pred, color = "g")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def prep(waves):
    df = pd.DataFrame()
    for i in range(len(waves[0].metricList)):
        tmp = []
        for j in range(len(waves)):
            for k in range(len(waves[j].assetList)):
                tmp.append(waves[j].assetList[k].abs_lift[waves[j].assetList[k].metricList[i]])

        df.insert(i,waves[0].metricList[i],tmp)
    return df


#RANDOM FOREST
def RandomForest(waves, predictorList, metric, control, user = True):
    if (user):
        features = aggregate_respondent_data(waves,control)
    else:
        features = aggregate_creative_data(waves, control)
    df = features
    labels = np.array(features[metric])
    tmp = pd.DataFrame()
    for i in range(len(predictorList)):
        tmp.insert(i,predictorList[i],df[predictorList[i]])
    features=tmp
    feature_list = list(features.columns)
    features = np.array(features)
    train_features, test_features, train_labels, test_labels = sklearn.model_selection.train_test_split(features, labels, test_size = 0.25, random_state = 42)
    rf = RandomForestRegressor(n_estimators = 1000)
    rf.fit(train_features, train_labels)
    # Get numerical feature importances
    importances = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    baseline = labels.mean()
    baseline_errors= []
    for i in range(len(test_labels)):
        baseline_errors.append((abs(baseline-test_labels[i])))
    baseline_errors = np.array(baseline_errors)
    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    print("\n\n\nRandom Forest:")
    # Print out the feature and importances
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
    print("Average baseline error: ", round(baseline_errors.mean(),4))
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 4), 'percentage points.')
    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')


def aggregate_respondent_data(waves,control):
    df = pd.DataFrame()
    for i in range(len(waves)):
        for j in range(len(waves[i].assetList)):
            if (control==True):
                tmp = waves[i].assetList[j].df_id
                if(df.empty==True):
                    df = tmp
                else:
                    new = df.append(tmp, ignore_index=True,sort=True)
                    df = new
            else:
                if(waves[i].assetList[j].id != 0):
                    tmp = waves[i].assetList[j].df_id
                    if(df.empty==True):
                        df = tmp
                    else:
                        new = df.append(tmp, ignore_index=True, sort=True)
                        df = new
    return df




def respondent_linear_regression(waves,x,y,control):
    print("\nRespondent-level Linear Regression:")
    df = aggregate_respondent_data(waves,control)
    var_y = df[y]
    var_x = df[x]
    model = sklearn.linear_model.LinearRegression()
    model.fit(var_x,var_y)
    for i in range(len(model.coef_)):
        print(x[i],": ",model.coef_[i])
    print("R2: ",model.score(var_x,var_y))



def respondent_linear_regression_count(waves,x,y,control):
    df = aggregate_respondent_data(waves,control)
    count_alter(df,x,1)
    var_y = df[y]
    var_x = df[['COUNT']]
    model = sklearn.linear_model.LinearRegression()
    model.fit(var_x,var_y)
    for i in range(len(model.coef_)):
        print("COUNT: ",model.coef_[i])
        print("Intercept: ",model.intercept_)
    print("R2: ",model.score(var_x,var_y))




def respondent_logistic_regression(waves, x,y,control):
    print("\n\nRespondent-level Logistic Regression:")
    df = aggregate_respondent_data(waves,control)
    model = sm.Logit(df[y],sm.add_constant(df[x]))
    result = model.fit()
    print(result.summary())
    print("Probability: ")
    print("Intercept: ",probability_intercept(result.params.values[0]))
    if (len(x) == 1):
        print(x[0], ": ",probability(result.params.values[0],result.params.values[1]))
    else:
        for i in range(len(x)):
            print(x[i], ": ",probability(result.params.values[0],result.params.values[i+1]))
    print("\n")




def correlation(waves,x,y,control,val=1):
    print("\n\nCorrelation Matrix:")
    df = aggregate_respondent_data(waves,control)
    if (type(x)==list):
        x1 = count_return(df,x,val)
        return np.corrcoef(np.asarray(df[y]),x1)
    else:
        return np.corrcoef(np.asarray(df[y]),np.asarray(df[x]))


def subset(df, keys):
    return df[keys]

def count_alter(data, x, val):
    array = np.asarray(subset(data,x))
    res = []
    for i in range(len(array)):
        count = 0
        for j in range(len(array[i])):
            if (array[i][j]==val):
                count+=1
        res.append(count)
    data.insert(len(data.columns),"COUNT",res)

def count_return(data,x,val):
    array = np.asarray(subset(data,x))
    res = []
    for i in range(len(array)):
        count = 0
        for j in range(len(array[i])):
            if (array[i][j]==val):
                count+=1
        res.append(count)
    return res

def dictionary(path):
    ref = pd.read_csv(path, index_col = False)
    refMetric = np.array(ref['Metric'])
    refValue = np.array(ref['Value'])
    mapDict = {}
    for i in range(len(refMetric)):
        mapDict[refMetric[i]]=refValue[i]
    return mapDict

def probability(b0,b1):
    y_pred = b0 + b1*(1)
    odds = math.exp(y_pred)
    p = odds/ (1+odds)
    return p

def probability_intercept(b):
    y_pred = b
    odds = math.exp(y_pred)
    p = odds / (1+odds)
    return p



if __name__ == '__main__':
    main()
