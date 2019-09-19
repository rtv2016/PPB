import numpy as np
import pandas as pd
from sklearn import metrics,neighbors

from old import visualize

"""post_process.py: Process predictions and get results
Example usage:
import prediction
predictions,actuals = prediction.main(); #Now use GUI to select appropriate files
post_process.main(predictions,actuals) #Just to display results
results,residuals = post_process.getResults(predictions,actuals) #To save results to variable

Changes from 0.0:
  Added first start at reliability index
  Added printing for Monte Cristo results
Future changes:
  Fully operation reliability indexer
"""
__author__    = "Brandon Veber"
__email__     = "veber001@umn.edu"
__version__   = "0.2"
__date__      = "7/28/2015"
__credits__   = ["Brandon Veber", "Rogelio Tornero-Velez", "Brandall Ingle",
                 "John Nichols"]
__status__    = "Development"

def MonteCarlo(results,verbose=0):
    arrayRes = {'Train':{'mae':[],'rmse':[]},'Drugs':{'mae':[],'rmse':[]},
                'Toxcast':{'mae':[],'rmse':[]}}
    for res in results:
        for key in res:
            for err in res[key]:
                arrayRes[key][err].append(res[key][err])
    print(np.mean(arrayRes['Toxcast']['mae']),np.std(arrayRes['Toxcast']['mae']),np.mean(arrayRes['Toxcast']['rmse']),np.std(arrayRes['Toxcast']['rmse']))
    if verbose > 0:
        print("\n%-28s%-28s%-28s"%('Training Error','Drugs-Test Error','Toxcast Error'))
        print("%-14s%-14s%-14s%-14s%-14s%-14s"%('MAE','RMSE','MAE','RMSE','MAE','RMSE'))
        print("%-7.4f%-7.4f%-7.4f%-7.4f%-7.4f%-7.4f%-7.4f%-7.4f%-7.4f%-7.4f%-7.4f%-7.4f"%
              (np.mean(arrayRes['Train']['mae']),np.std(arrayRes['Train']['mae']),np.mean(arrayRes['Train']['rmse']),np.std(arrayRes['Train']['rmse']),
               np.mean(arrayRes['Drugs']['mae']),np.std(arrayRes['Drugs']['mae']),np.mean(arrayRes['Drugs']['rmse']),np.std(arrayRes['Drugs']['rmse']),
               np.mean(arrayRes['Toxcast']['mae']),np.std(arrayRes['Toxcast']['mae']),np.mean(arrayRes['Toxcast']['rmse']),np.std(arrayRes['Toxcast']['rmse'])))

def main(preds,actuals,modelType,phase='I',plot=False,save=False,verbose=0):
    results,residuals = getResults(preds,actuals)
    if plot:
        visualize.plotHistogramResiduals(residuals)
    if save:
        writeResidualsToCSV(residuals,preds,actuals,modelType,phase)
    if verbose > 1:
        print(np.mean(drugs['y']),np.std(drugs['y']),np.min(drugs['y']),np.max(drugs['y']))
        print(np.mean(toxcast['y']),np.std(toxcast['y']),np.min(toxcast['y']),np.max(toxcast['y']))
    if verbose > 0:
        print("\n%-20s%-20s%-20s"%('Training Error','Drugs-Test Error','Toxcast Error'))
        print("%-10s%-10s%-10s%-10s%-10s%-10s"%('MAE','RMSE','MAE','RMSE','MAE','RMSE'))
        print("%-10.4f%-10.4f%-10.4f%-10.4f%-10.4f%-10.4f"%(results['Train']['mae'],results['Train']['rmse'],
                                                            results['Drugs']['mae'],results['Drugs']['rmse'],
                                                            results['Toxcast']['mae'],results['Toxcast']['rmse']))

def getResults(preds,actuals):
    """Calculates metrics
    Inputs:
      preds: dict, required.  The predictions for train, drugs-test and toxcast sets
      actuals: dict, required.  The real fraction unbound values for train, drugs-test and toxcast sets
    Outputs:
      results: dict.  The MAE and RMSE metrics for the train, drugs-test and toxcast sets
      residuals: dict.  The residual error for the train, drugs-test and toxcast sets
    """
    results = {'Train':{},'Drugs':{},'Toxcast':{}}
    results['Train']['rmse']=np.sqrt(metrics.mean_squared_error(preds['train'],actuals['train']))
    results['Train']['mae']=metrics.mean_absolute_error(preds['train'],actuals['train'])
    results['Drugs']['rmse']=np.sqrt(metrics.mean_squared_error(preds['drugs'],actuals['drugs']))
    results['Drugs']['mae']=metrics.mean_absolute_error(preds['drugs'],actuals['drugs'])
    results['Toxcast']['rmse']=np.sqrt(metrics.mean_squared_error(preds['toxcast'],actuals['toxcast']))
    results['Toxcast']['mae']=metrics.mean_absolute_error(preds['toxcast'],actuals['toxcast'])
    #Calculate residuals by calling getResiduals module
    residuals = getResiduals(preds,actuals)
    return(results,residuals)

def reliabilityIndex(train,test,preds,actuals):
    nbrs = neighbors.NearestNeighbors(n_neighbors=100,
                                               algorithm='ball_tree').fit(train['X'])
    distances, indices = nbrs.kneighbors(test['X'])
    reliability = []
    for i,sampleIndices in enumerate(indices):
        samplePreds = {'train':preds['train'][sampleIndices]}
        sampleActuals = {'train':actuals['train'][sampleIndices]}
        sampleResiduals = getResiduals(samplePreds,sampleActuals)
        reliability.append(sampleResiduals)#(np.mean(np.abs(sampleResiduals['train'])),
                            #np.std(sampleResiduals['train'])))
    return(reliability)

def gaussian(x,mu,sigma):
    return(np.exp(-np.power(x-mu,2.)/(2.*np.power(sigma,2.))))

def getResiduals(preds,actuals):
    """Calculates residual errors predictions - actuals
    Inputs:
      preds: dict, required. Keys should contain 'train', 'drugs', 'toxcast'
      actuals: dict, required. Keys should contain 'train', 'drugs', 'toxcast'
    Outputs:
      residuals: dict. Keys same as input dictionaries."""
    residuals = {}
    for key in preds:
        if key.split('_')[-1] != 'ind':
            residuals[key] = preds[key] - actuals[key]
    return(residuals)

def writeResidualsToCSV(residuals,preds,actuals,modelType,phase):
    trainIDs = getDrugIDs(preds['train_ind'])
    trainDF = makeDataFrame(trainIDs,residuals['train'],actuals['train'])
    trainDF.to_csv('Training_Residuals_'+modelType+'_Phase_'+str(phase)+'.csv')
    testIDs = getDrugIDs(preds['test_ind'])
    testDF = makeDataFrame(testIDs,residuals['drugs'],actuals['drugs'])
    testDF.to_csv('Drug_Test_Residuals_'+modelType+'_Phase_'+str(phase)+'.csv')
    #Toxcast: 3.0001-3.0238
    tox = pd.read_csv('C:/Users/Brandon/Documents/ORISE/toxcast_test_192_Phase_'+str(phase)+'.csv')
    toxIDs = np.array(tox['ID#'])
##    toxIDs = np.arange(3.0001,3.0239,.0001)
    toxDF = makeDataFrame(toxIDs,residuals['toxcast'],actuals['toxcast'])
    print('Here')
    toxDF.to_csv('Toxcast_Residuals_'+modelType+'_Phase_'+str(phase)+'.csv')
    return(trainIDs,testIDs)

def getDrugIDs(indices):
    #Drugs: 1.0001-1.0710, 2.0001-2.0022, 4.0001-4.0513
    IDs = []
    for ind in indices:
        if ind < 710: IDs.append(1+(ind+1)*1e-4)
        elif ind < (710+22): IDs.append(2+(ind-709)*1e-4)
        else: IDs.append(4+(ind-731)*1e-4)
    #IDs = np.array(["%.4f"%elem for elem in IDs]).reshape((len(IDs),1))
    return(IDs)

def makeDataFrame(IDs,residuals,actuals):
    IDs = np.array(["%.4f"%elem for elem in IDs]).reshape((len(IDs),1))
    residuals = residuals.reshape((len(residuals),1))
    actuals = actuals.reshape((len(actuals),1))
    df = pd.DataFrame(np.concatenate((IDs,residuals,actuals),axis=1),columns=['ID#','Error','FU'])
    df.set_index(['ID#'],inplace=True)
    return(df)

def calculateAIC(m,rmse,n):
    return(m*np.log(100*rmse**2)+2*(n+1))

def calculateBIC(m,rmse,n):
    return(m*np.log(100*rmse**2)+n*np.log(m))

def unscale(y,train,yscaler=None):
    if yscaler == 'lnKa':
        return(np.exp(2*y)/(1+np.exp(2*y)))#Equation for converting to fub from lnKa
    elif yscaler==None: return(y)
    else:
        #scaler = sklearn.preprocessing.StandardScaler().fit(train['y'])
        yScaled = yscaler.transform(y)
##        yScaled = sklearn.preprocessing.Imputer().fit_transform(yScaled)
        for i,y in enumerate(yScaled):
            if y>1: yScaled[i]=1
            elif y<0: yScaled[i] = 0
            else: yScaled[i] == y
        return(yScaled)
