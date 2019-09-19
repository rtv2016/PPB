import chem
from sklearn import metrics
import numpy as np
#import os, sys
import os
import pandas as pd

#modelType = 'KNN'  #KNN RF SVR
#featTypes = 'RF'   # SVR','RF','Lasso'
#nFeatures =  10     #None  # 10 for RF and SVM and 15 for KNN
#nSims = 1 #50
# xscale = None


#### KNN
modelType = 'KNN'
featTypes = 'RF'   # SVR','RF','Lasso'

## Reduced features from Ingle 2016, SI Table 1
descriptors = {"knn":["logS","logP(o/w)","SlogP","PEOE_VSA_FPPOS","GCUT_SMR_0",
                     "BCUT_SLOGP_0","GCUT_PEOE_0","GCUT_PEOE_1","GCUT_SLOGP_0",
                     "PEOE_VSA_PPOS"],
               "svr":["logS","VAdjEq","PEOE_VSA_FPPOS","SlogP","a_nS","a_base",
                      "a_nN","SlogP_VSA6","logP(o/w)","PEOE_VSA_FPOL"],
               "rf":["logS","logP(o/w)","SlogP","PEOE_VSA_FPPOS","GCUT_SMR_0",
                     "BCUT_SLOGP_0","GCUT_PEOE_0","GCUT_PEOE_1","GCUT_SLOGP_0",
                     "PEOE_VSA_PPOS"]}


path = os.path.dirname(__file__)
train, test = chem.Collector().collect()
## Reduce training set features to those found in Ingle 2016, SI Table 1
train['X'] = train['X'][descriptors[modelType.lower()]].copy()

## Reduce test set features to those found in Ingle 2016, SI Table 1
for key in test.keys():
    test[key]['X'] = test[key]['X'][descriptors[modelType.lower()]].copy()

scaler = chem.Scaler()
train_scaled = scaler.fit_transform(train)
test_scaled = {}
for key in test:
    test_scaled[key] = scaler.transform(test[key])
print("Features for modeling: ", train['X'].columns.values)

results = {}
for key in test_scaled:
    results[key] = []
for i in range(1):  # range(1)
    model = chem.Modeler(modelType='KNN', random_state=i, verbose=1).fit(train_scaled['X'], train_scaled['y'])
    for key in test_scaled:
        kNN_preds = model.predict(test_scaled[key]['X'])
        mae = metrics.mean_absolute_error(scaler.yscaler.inverse_transform(kNN_preds), test[key]['y'])
        results[key].append(mae)
for key in results:
    print(key, np.mean(results[key]), np.std(results[key]))

#### SVR
modelType = 'SVR'
featTypes = 'SVR'   # SVR','RF','Lasso'


## Reduced features from Ingle 2016, SI Table 1
descriptors = {"knn":["logS","logP(o/w)","SlogP","PEOE_VSA_FPPOS","GCUT_SMR_0",
                     "BCUT_SLOGP_0","GCUT_PEOE_0","GCUT_PEOE_1","GCUT_SLOGP_0",
                     "PEOE_VSA_PPOS"],
               "svr":["logS","VAdjEq","PEOE_VSA_FPPOS","SlogP","a_nS","a_base",
                      "a_nN","SlogP_VSA6","logP(o/w)","PEOE_VSA_FPOL"],
               "rf":["logS","logP(o/w)","SlogP","PEOE_VSA_FPPOS","GCUT_SMR_0",
                     "BCUT_SLOGP_0","GCUT_PEOE_0","GCUT_PEOE_1","GCUT_SLOGP_0",
                     "PEOE_VSA_PPOS"]}


path = os.path.dirname(__file__)
train, test = chem.Collector().collect()
## Reduce training set features to those found in Ingle 2016, SI Table 1
train['X'] = train['X'][descriptors[modelType.lower()]].copy()

## Reduce test set features to those found in Ingle 2016, SI Table 1
for key in test.keys():
    test[key]['X'] = test[key]['X'][descriptors[modelType.lower()]].copy()

scaler = chem.Scaler()
train_scaled = scaler.fit_transform(train)
test_scaled = {}
for key in test:
    test_scaled[key] = scaler.transform(test[key])
print("Features for modeling: ", train['X'].columns.values)

results = {}
for key in test_scaled:
    results[key] = []
for i in range(1):  # range(1)
    model = chem.Modeler(modelType='SVR', random_state=i, verbose=1).fit(train_scaled['X'], train_scaled['y'])
    for key in test_scaled:
        SVR_preds = model.predict(test_scaled[key]['X'])
        mae = metrics.mean_absolute_error(scaler.yscaler.inverse_transform(SVR_preds), test[key]['y'])
        results[key].append(mae)
for key in results:
    print(key, np.mean(results[key]), np.std(results[key]))



#### RF
modelType = 'RF'
featTypes = 'RF'   # SVR','RF','Lasso'

## Reduced features from Ingle 2016, SI Table 1
descriptors = {"knn":["logS","logP(o/w)","SlogP","PEOE_VSA_FPPOS","GCUT_SMR_0",
                     "BCUT_SLOGP_0","GCUT_PEOE_0","GCUT_PEOE_1","GCUT_SLOGP_0",
                     "PEOE_VSA_PPOS"],
               "svr":["logS","VAdjEq","PEOE_VSA_FPPOS","SlogP","a_nS","a_base",
                      "a_nN","SlogP_VSA6","logP(o/w)","PEOE_VSA_FPOL"],
               "rf":["logS","logP(o/w)","SlogP","PEOE_VSA_FPPOS","GCUT_SMR_0",
                     "BCUT_SLOGP_0","GCUT_PEOE_0","GCUT_PEOE_1","GCUT_SLOGP_0",
                     "PEOE_VSA_PPOS"]}


path = os.path.dirname(__file__)
train, test = chem.Collector().collect()
## Reduce training set features to those found in Ingle 2016, SI Table 1
train['X'] = train['X'][descriptors[modelType.lower()]].copy()

## Reduce test set features to those found in Ingle 2016, SI Table 1
for key in test.keys():
    test[key]['X'] = test[key]['X'][descriptors[modelType.lower()]].copy()

scaler = chem.Scaler()
train_scaled = scaler.fit_transform(train)
test_scaled = {}
for key in test:
    test_scaled[key] = scaler.transform(test[key])
print("Features for modeling: ", train['X'].columns.values)

results = {}
for key in test_scaled:
    results[key] = []
for i in range(1):  # range(1)
    model = chem.Modeler(modelType='RF', random_state=i, verbose=1).fit(train_scaled['X'], train_scaled['y'])
    for key in test_scaled:
        RF_preds = model.predict(test_scaled[key]['X'])
        mae = metrics.mean_absolute_error(scaler.yscaler.inverse_transform(RF_preds), test[key]['y'])
        results[key].append(mae)
for key in results:
    print(key, np.mean(results[key]), np.std(results[key]))


fup_KNN = np.exp(2*kNN_preds)/(1+np.exp(2*kNN_preds))
fup_RF = np.exp(2*RF_preds)/(1+np.exp(2*RF_preds))
fup_SVR = np.exp(2*SVR_preds)/(1+np.exp(2*SVR_preds))


combine = {'kNN_py_pred': fup_KNN, 'SVM_py_pred': fup_SVR, 'RF_py_pred': fup_RF }
doutput = pd.DataFrame(data=combine)

file_name ='ToxCast_2_qc.csv'
doutput.to_csv(file_name)




