import sys

import numpy as np
import sklearn

sys.path.append('C:/Users/Brandon/Desktop')
sys.path.append('C:/Users/Brandon/Documents/ORISE/')
from old import post_process, preprocess, data_collection

"""predict.py: Model evaluation and prediction
Example usage:
import data_collection as dc
import preprocess, predict
drugs,toxcast = dc.getData(None,None); #Now use GUI to select appropriate files
train,test,toxcast = preprocess.main(drugs,toxcast)
predictions,actual_values = predict.main()

Changes for version_0.0:
  updated SVM grid search parameters
  removed grid search with RF, only set n_estimators = 300
  added MonteCristo style simulation
"""
__author__    = "Brandon Veber"
__email__     = "veber001@umn.edu"
__version__   = "0.1"
__date__      = "7/25/2015"
__credits__   = ["Brandon Veber", "Rogelio Tornero-Velez", "Brandall Ingle",
                 "John Nichols"]
__status__    = "Development"
        

def main(trainingFile,testFile,toxcastFile,
        featSelect='predefined_RF',nFeatures=10,phase=1,plot=False,save=False,random_state=1,
         feat_random_state=1,modelType='RF',numTrainingSamples=1045,preSplit=True,
         verbose=0):
    """Main machine learning analysis module.  Collects, and scales all data.  Then makes predictions
    and calculates result metrics
    Inputs:
      featSelect: string, optional (default='drugs'). 
      nFeatures: int, optional (default=20). The number of features to keep.
      plot: bool, optional (default=False). If True, histograms of residuals are plotted and saved
      save: bool, optional (default=False). If True, residual errors are written to CSV files
      verbose: int, optional (default=0). The verbosity of output statement      
    """
    #Hard-coded variables
    xScale='MinMax' #scale type for the descriptors. Supported options are MinMax and standard
    yScale='lnKa' #scale type for the fraction unbound target values. Supported options are the lnKa (pseudo equibilibrium) and None
    if phase==1:phase='I'
    else: phase='II'
    #Preprocessing
    if preSplit:
        #trainingFile = 'C:/Users/Brandon/Documents/ORISE/drug_training_192.csv'
        #testFile = 'C:/Users/Brandon/Documents/ORISE/drug_test_192.csv'
        #toxcastFile = 'C:/Users/Brandon/Documents/ORISE/toxcast_test_192_Phase_'+phase+'.csv'#toxcast_test_192.csv'
        train,test,toxcast,yscaler = preprocess.mainPreSplit(trainingFile,testFile,toxcastFile,
                                                     featSelect,modelType,random_state,feat_random_state,
                                                     yScale,xScale,nFeatures,numTrainingSamples,
                                                     verbose)
    else:
        drugs,toxcast = data_collection.getData()
        train,test,toxcast,yscaler = preprocess.main(drugs,toxcast,featSelect,modelType,
                                                     random_state, feat_random_state,yScale,
                                                     xScale,nFeatures,numTrainingSamples,verbose)
    #Model Creation
    estimator = getEstimator(modelType)
    modelParams = getModelParams(modelType,estimator)
    if modelType in ['RF']:
        clf = getEstimator(modelType)
    else:
        clf = sklearn.grid_search.GridSearchCV(estimator,modelParams,#fit_params={'sample_weight':weights},
                                   scoring='mean_squared_error',cv=3,verbose=verbose,n_jobs=-1)
    #Find error dependant on y scaling
    predsTrain = post_process.unscale(sklearn.cross_validation.cross_val_predict(clf,train['X'],train['y_scaled'],cv=5),train,'lnKa')
    clf.fit(train['X'],train['y_scaled'])
##    predsTrain = getPredictions(clf,train,train,yscaler=yscaler)
    predsDrugs = getPredictions(clf,train,test,yscaler=yscaler)
    predsToxcast = getPredictions(clf,train,toxcast,yscaler=yscaler)
    if verbose > 0 and modelType not in ['RF']:
        print('Best estimator: ',clf.best_estimator_)
    #Create dictionaries for return variables
    preds = {'train':predsTrain,'train_ind':train['indices'],
             'drugs':predsDrugs,'test_ind':test['indices'],'toxcast':predsToxcast}
    actuals = {'train':train['y'],'drugs':test['y'],'toxcast':toxcast['y']}
    post_process.main(preds,actuals,modelType,phase,plot=plot,save=save,verbose=verbose)
    return(preds,actuals)#,train,test,toxcast)

def getPredictions(clf,train,test,modelType='svm',gridSearch=True,pipe=None,yscaler=None,verbose=0):
    if gridSearch and verbose > 0: 
        print(clf.best_estimator_)
    elif verbose > 0:
        print(clf)
    if modelType in ['knnC','lr','cluster']:
        preds = clf.predict_proba(test['X'])[:,1]
    else: preds = clf.predict(test['X'])
    if yscaler: preds = post_process.unscale(preds,train,yscaler)
    return(preds)

def getEstimator(modelType):
    if modelType in ['libsvm_rbf','libsvm_lin','SVR']:
        estimator = sklearn.svm.SVR()
    if modelType == 'KNN':
        estimator = sklearn.neighbors.KNeighborsRegressor()
    if modelType == 'RF':
        estimator = sklearn.ensemble.RandomForestRegressor(n_estimators=100)
    if modelType == 'Adaboost':
        estimator = sklearn.ensemble.AdaBoostRegressor()
        
    return(estimator)

def getModelParams(modelType,estimator):
    modelParams = None
    if modelType == 'SVR':
        modelParams = {'kernel':['rbf'],'C':[10,50],
            'epsilon':np.logspace(-1,0,3),#,'gamma':[.1,.25],#np.arange(.02,.15,.02),#np.logspace(-2,0,3)
            'cache_size':[4096]}
    if modelType =='KNN':        
        modelParams = {'n_neighbors':[2,10,25],'leaf_size':[1,10],#'weights':['uniform','distance'],
                    'algorithm':['auto','ball_tree']}#,
    if modelType == 'RF':
        modelParams = {'n_estimators':[500],'n_jobs':[-1],
                    'max_features':['auto'],'min_samples_split':[50]}
    if modelType == 'Adaboost':
        modelParams = {'n_estimators':[25,50,75,100],'learning_rate':np.arange(.05,.15,.02),
                    'loss':['linear','square','exponential']}#,
    return(modelParams)


# def MonteCarlo(nSims=25,featSelect='drugs',nFeatures=10,plot=False,save=False,
#                 random_state=1,modelType='SVR',numTrainingSamples=1045,preSplit=False,
#                 verbose=0):
#     res = []
#     for i in range(nSims):
#         if verbose > 0:print('\nMonte Carlo Simulation: ', i+1)
#         preds,actuals = main(featSelect,nFeatures,plot,save,random_state,
#                              i,modelType,numTrainingSamples,
#                              preSplit,verbose)
#         results,residuals = post_process.getResults(preds,actuals)
#         res.append(results)
#     try:
#         post_process.MonteCarlo(res,verbose)
#         return(res)
#     except:
#         return(res)
#
# def foldValidation(nFeatures=10,nFolds=10):
#     trainingFile = 'C:/Users/Brandon/Documents/ORISE/drug_training_192.csv'
#     testFile = 'C:/Users/Brandon/Documents/ORISE/drug_test_192.csv'
#     toxcastFile = 'C:/Users/Brandon/Documents/ORISE/toxcast_test_192.csv'
#     train,test,toxcast,yscaler = preprocess.mainPreSplit(trainingFile,testFile,
#                                                          toxcastFile,nFeatures=192)
#     kfTrain = sklearn.cross_validation.KFold(len(train['y']),nFolds)
#     modelOrig = sklearn.ensemble.RandomForestRegressor(n_estimators=100,random_state=1)
#     model = sklearn.ensemble.RandomForestRegressor(n_estimators=100)
#     #modelOrig.fit(train['X'],train['y'])
#     featuresListOrig = [88, 89, 154, 123, 73, 32, 65, 66, 69, 108, 129]#np.sort(np.argsort(modelOrig.feature_importances_)[-nFeatures:])
#     print('Original Feature List\n',featuresListOrig)
#     modelOrig.fit(train['X'][:,featuresListOrig],train['y_scaled'])
#     drugTrainRes = {'Train':{'mae':[],'rmse':[]},'Drugs':{'mae':[],'rmse':[]},'Toxcast':{'mae':[],'rmse':[]}}
#     kfTox = sklearn.cross_validation.KFold(len(toxcast['y']),nFolds)
#     res = {'Train':{'mae':[],'rmse':[]},'Drugs':{'mae':[],'rmse':[]},'Toxcast':{'mae':[],'rmse':[]}}
#     toxTrainRes = {'Train':{'mae':[],'rmse':[]},'Drugs':{'mae':[],'rmse':[]},'Toxcast':{'mae':[],'rmse':[]}}
#     X = {'Train':train['X'],'Drugs':test['X'],'Toxcast':[]}
#     actuals={'Train':train['y'],'Drugs':test['y'],'Toxcast':[]}
#     for trainInd,testInd in kfTox:
#         X_train,X_test = toxcast['X'][trainInd],toxcast['X'][testInd]
#         y_train,y_test = toxcast['y_scaled'][trainInd],toxcast['y'][testInd]
#         model.fit(X_train,y_train)
#         featuresList = np.sort(np.argsort(model.feature_importances_)[-nFeatures:])
#         print(featuresList)
#         model.fit(X_train[:,featuresList],y_train)
#         X['Toxcast'] = X_test
#         actuals['Toxcast'] = y_test
#         for key in toxTrainRes:
#             predsOrig = post_process.unscale(modelOrig.predict(X[key][:,featuresListOrig]),train,yscaler)
#             preds = post_process.unscale(model.predict(X[key][:,featuresList]),train,yscaler)
#             drugTrainRes[key]['mae'].append(sklearn.metrics.mean_absolute_error(predsOrig,actuals[key]))
#             drugTrainRes[key]['rmse'].append(np.sqrt(sklearn.metrics.mean_squared_error(predsOrig,actuals[key])))
#             toxTrainRes[key]['mae'].append(sklearn.metrics.mean_absolute_error(preds,actuals[key]))
#             toxTrainRes[key]['rmse'].append(np.sqrt(sklearn.metrics.mean_squared_error(preds,actuals[key])))
#     d = drugTrainRes; t = toxTrainRes
#     for key in t:
#         print(key);print('Drugs Training')
#         print(np.mean(d[key]['mae']),np.std(d[key]['mae']));
#         print(np.mean(d[key]['rmse']),np.std(d[key]['rmse']))
#         print('Toxcast Training')
#         print(np.mean(t[key]['mae']),np.std(t[key]['mae']));
#         print(np.mean(t[key]['rmse']),np.std(t[key]['rmse']))
#     return(drugTrainRes,toxTrainRes)

