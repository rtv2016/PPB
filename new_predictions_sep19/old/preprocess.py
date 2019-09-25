import sys
import random

import numpy as np
import sklearn
from sklearn import model_selection, ensemble, feature_selection

from old import data_collection


"""preprocess.py: Split training and test sets, and scale values
Example usage:
import data_collection as dc
import preprocess
drugs,toxcast = dc.getData(None,None); #Now use GUI to select appropriate files
train,test,toxcast = preprocess.main(drugs,toxcast)

Changes from 0.0:
  Feature elimination on half of training set to reduce overfitting
  Normalize datasets prior to feature selection
  If modelType not in ['RF','SVR'] default feature selection set to SVR
  More estimators for RF feature selection
Changes from 0.1
  Added normalize2 which simplifies scaling to a 1 training set and 1 test set
Future changes:
  Change all preprocessing to 1 training set and 1 test set
  Create lnKa scaler class to use with sklearn
"""
__author__    = "Brandon Veber"
__email__     = "veber001@umn.edu"
__version__   = "0.2"
__date__      = "7/28/2015"
__credits__   = ["Brandon Veber", "Rogelio Tornero-Velez", "Brandall Ingle",
                 "John Nichols"]
__status__    = "Development"


def main(drugs,toxcast,featSelect='drugs',featTypes='SVR',random_state=1,
         feat_random_state=4,yscale='lnKa',xscale='MinMax',nFeatures=20,
         numTrainingSamples=1045,verbose=0):
    """Splits data and makes predictions
    Inputs:
      drugs: dict, required. 'X' key is input features,'y' key is fraction unbound target value
      toxcast: dict, required. 'X' key is input features,'y' key is fraction unbound target value
      featSelect: string, optional (default='drugs'). 
      random_state: int, optional (default=1). Seed for random number generator
      yscale: str, optional (default 'lnKa').  This is the scale type for the fraction unbound target values.
              Supported options are the lnKa (pseudo equibilibrium) and None
      xscale: str, optional (default 'MinMax'). This is the scale type for the descriptors.
              Supported options are MinMax and standard
      nFeatures: int, optional (default=20). The number of features to keep.
      modelType: string, optional (default='SVR'). The selectable machine learing algorithm.
      nTrainingSamples: int, optional (default=1045). The number of training samples to use.
      verbose: int, optional (default=0). The verbosity of output statement
    Outputs:
      train: dict.
      test: dict.
      toxcastC: dict.
    """
    #Create copy of features and target values
    drugsC = {'y':np.array(drugs['y']),'X':np.array(drugs['X']).copy()}
    toxcastC = {'y':np.array(toxcast['y']),'X':np.array(toxcast['X']).copy()}
    #Split drug set into train and test
    train,test = getTrainTest(drugsC,numTrainingSamples,random_state)
    #Scale input features
    train,test,toxcastC,yscaler = normalize(train,test,toxcastC,xscale,yscale)
    #Find important features
    featureList = findFeatures(train,nFeatures,featSelect,featTypes=featTypes,
                               feat_random_state=feat_random_state)
    if verbose > 0: print(featureList)
    #Reduce the data sets
    train['X'] = train['X'][:,featureList]
    test['X'] = test['X'][:,featureList]
    toxcastC['X'] = toxcastC['X'][:,featureList]
    #Split drug set into train and test
    trainIon = data_collection.getIonicClass('ABNZ_Ionic_DrugTraining.xlsx')
    drugTestIon = data_collection.getIonicClass('ABNZ_Ionic_DrugTest.xlsx')
    toxIon = data_collection.getIonicClass('ABNZ_Ionic_ToxCastTest.xlsx')
    return(train,test,toxcastC,yscaler)


def mainPreSplit(trainingFile,testFile,toxcastFile,featSelect='drugs',featTypes='SVR',
                 random_state=1,feat_random_state=4,yscale='lnKa',
                 xscale='MinMax',nFeatures=20,numTrainingSamples=1045,verbose=0):
    
    train = data_collection.extractCSV(trainingFile,'MOE')
    test = data_collection.extractCSV(testFile,'MOE')
    toxcast = data_collection.extractCSV(toxcastFile,'MOE')
    random.seed(random_state)
    trainIndices = random.sample(range(len(train['y'])),numTrainingSamples)
    train['indices']=trainIndices;test['indices']=range(len(test['y']));toxcast['indices']=range(len(toxcast['y']))
    train['X'] = np.array(train['X'])[trainIndices];train['y']=train['y'][trainIndices]
    #Scale input features
    train,test,toxcast,yscaler = normalize(train,test,toxcast,xscale,yscale)
    #Find important features
    featureList = findFeatures(train,nFeatures,featSelect,featTypes=featTypes,
                               feat_random_state=feat_random_state)
    if verbose > 0: print(featureList)
    #Reduce the data sets
    train['X'] = np.array(train['X'])[:,featureList]
    test['X'] = np.array(test['X'])[:,featureList]
    toxcast['X'] = np.array(toxcast['X'])[:,featureList]
    return(train,test,toxcast,yscaler)

def findFeatures(data,nFeatures=10,featSelect='drugs',
                 featTypes='RF',feat_random_state=4):
    """Find the top features for a given dataset using LinearSVR Lasso and Random Forest
    Inputs
      data: matrix, n x d matrix (n = number of samples; d = number of features). Required
      nFeatures: int, number of features to keep. Optional (default = 25)
    Outputs
      featureList: list, n top features for each machine learning algorithm
    """
    #Transform data to numpy array
    if featSelect.split('_')[0] == 'predefined':
        return(subFeatures(featSelect.split('_')[1]))
    elif featSelect == 'drugs':
        #Initialize Dictionaries
        clf,topFeatures = {},{}
        endIndex = int(len(data['y'])/2)
        random.seed(feat_random_state)
        ind = random.sample(range(len(data['y'])),len(data['y']))
        X = data['X'][ind]
        y = data['y'][ind]
        if featTypes not in ['SVR','RF','Lasso']: featTypes = 'RF'
        #if featTypes == 'SVR': featTypes = 'RF'
        #Recursive Feature Elimination for SVR
        if 'SVR' in featTypes:
            #orig: 4

            grid = model_selection.GridSearchCV(sklearn.svm.LinearSVR(random_state=feat_random_state),
                                            {'C':np.logspace(-2,0,3),'epsilon':np.logspace(-2,0,3)#,
                                             #'loss':['epsilon_insensitive','squared_epsilon_insensitive']
                                             #}).fit(X[:endIndex],y[:endIndex])
                                            }).fit(X[:endIndex],y[:endIndex])
            #model = grid.best_estimator_;print(model)
            model = sklearn.svm.LinearSVR(C=.1,random_state=feat_random_state)
            clf['SVR'] = sklearn.feature_selection.RFE(model,nFeatures,10).fit(X[:endIndex],y[:endIndex])
##            clf['SVR'] = sklearn.feature_selection.RFE(sklearn.svm.LinearSVR(random_state=feat_random_state),
##                                                   nFeatures,10).fit(data['X'][:endIndex],data['y'][:endIndex])
            #Indices of top SVR features
            return(np.where(clf['SVR'].ranking_ == 1)[0])
        #Recursive Feature Elimination for Lasso
        if 'Lasso' in featTypes:
            clf['Lasso'] = sklearn.feature_selection.RFE(sklearn.linear_model.Lasso(alpha=.001),
                                                   nFeatures).fit(X[:endIndex],y[:endIndex])
            #Indices of top Lasso features
            return(np.where(clf['Lasso'].ranking_ == 1)[0])
        #Random Forest
        if 'RF' in featTypes:
            clf['RF'] = ensemble.RandomForestRegressor(random_state=feat_random_state,
                                                               n_estimators=100).fit(X[:endIndex],
                                                                                        y[:endIndex])
            #Indices of top RF features
##            print(np.argsort(clf['RF'].feature_importances_)[-nFeatures:])
            return(np.argsort(clf['RF'].feature_importances_)[-nFeatures:])
##            return(np.sort(np.argsort(clf['RF'].feature_importances_)[-nFeatures:]))
    elif featSelect == None:
        return(range(len(data['X'][0])))
    else:
        print('Incorrect feature selection type')
        
def subFeatures(modelType):
    #featureList = 'a_donacc BCUT_SLOGP_0 BCUT_SLOGP_2 BCUT_SMR_2 BCUT_SMR_3 GCUT_PEOE_0 GCUT_SLOGP_0 GCUT_SLOGP_1 GCUT_SLOGP_2 logS PEOE_RPC- PEOE_VSA+0 PEOE_VSA+1 PEOE_VSA+3 PEOE_VSA-3 PEOE_VSA-4 PEOE_VSA_FNEG SlogP_VSA7 SMR_VSA2 SMR_VSA4'.split(' ')
    #featuresList = [65,73,87,175]#['GCUT_SLOGP_0', 'logS', 'PEOE_RPC-', 'PEOE_VSA+1']
    #featuresList = [15, 22, 29, 31, 33, 35, 56, 65, 66, 68, 69, 73, 87, 89, 94, 103, 120, 175,176, 178]
    if modelType == 'SVR': featuresList = [89, 175, 123, 154, 26, 8, 23, 161, 88, 121]
    else: featuresList = [89, 88, 154, 123, 73, 32, 65, 66, 69, 129]
    
    #featureList = ['logS']
    return(featuresList)

def getTrainTest(data,numTrainingSamples=1045,random_state=1):
    """Splits drugs into train and test set
    Inputs:
      data: dict, required. Contains all drug data
      numTrainingSamples: int, optional (default=1045). Number of training samples
      random_state: int, optional (default=1). Seed for random nmber generator
    Outputs:
      train: dict. The training samples, 'X' key is input features,
             'y' key is fraction unbound target value
      test: dict. The test samples, 'X' key is input features,
             'y' key is fraction unbound target value"""
    #Correct invalid number of training samples
    if numTrainingSamples > 1045:
        print('Too many training samples requested, converting to 1045')
        numTrainingSamples = 1045
    elif numTrainingSamples < 3:
        print('nSamples < nFolds for model selection, converting to 3')
        numTrainingSamples=3
    #Hard coded number of test samples
    numTest = 200
    #Random number initialization
    random.seed(random_state)
    #Get indices of training and test sets
    testIndices = random.sample(range(len(data['y'])),numTest)
    trainIndices = np.array(list(set(range(len(data['y'])))-set(testIndices)))
    if random_state !=1: np.random.shuffle(trainIndices)
    trainIndices = trainIndices[:numTrainingSamples]
    #Create dictionaries for return variables
    train = {'X':data['X'][trainIndices],'y':data['y'][trainIndices],'indices':trainIndices}
    test = {'X':data['X'][testIndices],'y':data['y'][testIndices],'indices':testIndices}
    return(train,test)

def normalize(train,test,toxcast,xscale='MinMax',yscale='lnKa'):
    """Scales all input features
    Inputs:
      train: dict, required. The training data
      test: dict, required. The drugs-test data
      toxcast: dict, required. The Toxcast data
      xscale: str, optional (default 'MinMax'). This is the scale type for the descriptors.
              Supported options are MinMax and standard
      yscale: str, optional (default 'lnKa').  This is the scale type for the fraction unbound target values.
              Supported options are the lnKa (pseudo equibilibrium) and None
    Outputs:
      train_scaled: dict. The scaled training data
      test_scaled: dict. The scaled drugs-test data
      toxcast_scaled: dict. The scaled Toxcast data
    """
    #Initialize output variables
    train_scaled = {'y':train['y'],'indices':train['indices']}
    test_scaled= {'y':test['y'],'indices':test['indices']}
    toxcast_scaled = {'y':toxcast['y']}
    #Create imputer to replace missing values and transform test data
    imputer = sklearn.preprocessing.Imputer()
    train['X'] = imputer.fit_transform(train['X'])
    test['X'] = imputer.transform(test['X'])
    toxcast['X'] = imputer.transform(toxcast['X'])
    #Create X scaler
    if xscale=='MinMax':
        scaler = sklearn.preprocessing.MinMaxScaler()
    elif xscale=='standard':
        scaler = sklearn.preprocessing.StandardScaler()
    #Fit scaler to training data, and transform training data
    train_scaled['X'] = scaler.fit_transform(train['X'])
    #Transform two test sets
    test_scaled['X'] = scaler.transform(test['X'])
    toxcast_scaled['X'] = scaler.transform(toxcast['X'])
    #Create y scaler and transform y values
    if yscale=='lnKa':
        yscaler = 'lnKa'
        #lnKa pseudo-equilibrium transformation of fraction unbound value
        train_scaled['y_scaled'] = lnKaScaler(train['y'])
        test_scaled['y_scaled'] = lnKaScaler(test['y'])
        toxcast_scaled['y_scaled'] = lnKaScaler(toxcast['y'])
    elif yscale=='standard':
        yscaler = sklearn.preprocessing.StandardScaler()
        train_scaled['y_scaled'] = yscaler.fit_transform(train['y'])
        test_scaled['y_scaled'] = yscaler.transform(test['y'])
        toxcast_scaled['y_scaled'] = yscaler.transform(toxcast['y'])
    return(train_scaled,test_scaled,toxcast_scaled,yscaler)

def normalize2(train,test,xscale='MinMax',yscale='lnKa'):
    """Scales all input features
    Inputs:
      train: dict, required. The training data
      test: dict, required. The test data
      xscale: str, optional (default 'MinMax'). This is the scale type for the descriptors.
              Supported options are MinMax and standard
      yscale: str, optional (default 'lnKa').  This is the scale type for the fraction unbound target values.
              Supported options are the lnKa (pseudo equibilibrium) and None
    Outputs:
      train_scaled: dict. The scaled training data
      test_scaled: dict. The scaled drugs-test data
    """
    #Initialize output variables
    train_scaled = {'y':train['y']}
    test_scaled= {'y':test['y']}
    if 'indices' in list(train.keys()):train_scaled['indices']=train['indices']
    if 'indices' in list(test.keys()):test_scaled['indices']=test['indices']
##    toxcast_scaled = {'y':toxcast['y']}
    #Create imputer to replace missing values and transform test data
    imputer = sklearn.preprocessing.Imputer()
    train['X'] = imputer.fit_transform(train['X'])
    test['X'] = imputer.transform(test['X'])
##    toxcast['X'] = imputer.transform(toxcast['X'])
    #Create X scaler
    if xscale=='MinMax':
        scaler = sklearn.preprocessing.MinMaxScaler()
    elif xscale=='standard':
        scaler = sklearn.preprocessing.StandardScaler()
    #Fit scaler to training data, and transform training data
    train_scaled['X'] = scaler.fit_transform(train['X'])
    #Transform two test sets
    test_scaled['X'] = scaler.transform(test['X'])
##    toxcast_scaled['X'] = scaler.transform(toxcast['X'])
    #Create y scaler and transform y values
    if yscale=='lnKa':
        yscaler = 'lnKa'
        #lnKa pseudo-equilibrium transformation of fraction unbound value
        train_scaled['y_scaled'] = lnKaScaler(train['y'])
        test_scaled['y_scaled'] = lnKaScaler(test['y'])
##        toxcast_scaled['y_scaled'] = lnKaScaler(toxcast['y'])
    elif yscale=='standard':
        yscaler = sklearn.preprocessing.StandardScaler()
        train_scaled['y_scaled'] = yscaler.fit_transform(train['y'])
        test_scaled['y_scaled'] = yscaler.transform(test['y'])
##        toxcast_scaled['y_scaled'] = yscaler.transform(toxcast['y'])
    return(train_scaled,test_scaled,yscaler)#,toxcast_scaled,yscaler)

def lnKaScaler(values):
    """Scale fraction unbound to pseudo-equilibrium constant
    Inputs:
      values: array, required. The array of fraction unbound values
    Outputs:
      The array of fraction unbound values
    """
    y_temp = np.array([np.max([.001,np.min([.99,sample])]) for sample in values])
    return(.5*np.log(y_temp/(1-y_temp)))

def reduceByIonicClass(data,ion,ionicClass):
    indices = np.where((ion=='N')|(ion=='A')|(ion=='Z'))[0]
    newData = {}
    print('N & A & Z')
    for key in data:
        newData[key] = np.array(data[key])[indices]
    return(newData)
