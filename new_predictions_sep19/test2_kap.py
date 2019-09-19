import chem
from sklearn import metrics
import numpy as np
import os, sys
import pandas as pd

modelType = 'KNN'  #KNN RF SVR
featTypes = 'RF'   # SVR','RF','Lasso'
nFeatures =  15     #None  # 10 for RF and SVM and 15 for KNN
nSims = 1 #50
# xscale = None

## Reduced features from Ingle 2016, SI Table 1
descriptors = {"knn":["PEOE_VSA_FPPOS","SlogP","logS","logP(o/w)","GCUT_SMR_0",
                      "BCUT_SLOGP_0","GCUT_PEOE_0","GCUT_SLOGP_0","GCUT_PEOE_1",
                      "BCUT_SLOGP_2","PEOE_VSA+1","PEOE_VSA+4","PEOE_VSA_PPOS",
                      "PEOE_VSA_POL","SMR_VSA6"],
               "svr":["logS","VAdjEq","PEOE_VSA_FPPOS","SlogP","a_nS","a_base",
                      "a_nN","SlogP_VSA6","logP(o/w)","PEOE_VSA_FPOL"],
               "rf":["logS","logP(o/w)","SlogP","PEOE_VSA_FPPOS","GCUT_SMR_0",
                     "BCUT_SLOGP_0","GCUT_PEOE_0","GCUT_PEOE_1","GCUT_SLOGP_0",
                     "PEOE_VSA_PPOS"]}


path = os.path.dirname(__file__)
# train, test = chem.Collector(os.path.join(path, 'data', 'toxcast_test_192_Phase_I.csv'),
#                              os.path.join(path, 'data', 'toxcast_test_192_Phase_II.csv')).collect()
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
## chem.Reducer is used to reduce number of features according featTypes method
# reducer = chem.Reducer(yscaler=scaler.yscaler, nFeatures=nFeatures, featSelect='drugs', featTypes=featTypes,
#                        modelType=modelType, nSims=nSims, verbose=2)
# train_scaled_reduced = reducer.fit_transform(train_scaled)
# print(train['X'].columns.values[reducer.featureList], reducer.featureList)
# print(reducer.featureRank)
# test_scaled_reduced = {}
# for key in test:
#     test_scaled_reduced[key] = reducer.transform(test_scaled[key])
results = {}
for key in test_scaled:
    results[key] = []
for i in range(1):  # range(1)
    model = chem.Modeler(modelType, random_state=i, verbose=1).fit(train_scaled['X'], train_scaled['y'])
    for key in test_scaled:
        preds = model.predict(test_scaled[key]['X'])
        mae = metrics.mean_absolute_error(scaler.yscaler.inverse_transform(preds), test[key]['y'])
        results[key].append(mae)
for key in results:
    print(key, np.mean(results[key]), np.std(results[key]))


# for key in test_scaled_reduced:
#     results[key] = []
# for i in range(1):  # range(1)
#     model = chem.Modeler(modelType, random_state=i, verbose=1).fit(train_scaled_reduced, train_scaled['y'])
#     for key in test_scaled_reduced:
#         preds = model.predict(test_scaled_reduced[key])
#         mae = metrics.mean_absolute_error(scaler.yscaler.inverse_transform(preds), test[key]['y'])
#         results[key].append(mae)
# for key in results:
#     print(key, np.mean(results[key]), np.std(results[key]))


fup = np.exp(2*preds)/(1+np.exp(2*preds))
dfup = pd.DataFrame(fup)
file_name = modelType + '_' + featTypes + '_Feats' + str(nFeatures)+'Auto_Leaf_size=30.csv'
dfup.to_csv(file_name)

#np.savetxt(modelType + ' '+ featTypes + ' NoFeats= '+ str(nFeatures)+ ' Leafsize=2', fup)
