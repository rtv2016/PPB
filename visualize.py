import pandas as pd
import numpy as np
import sklearn
import sys
#sys.path.append('C:/Users/Brandon/Desktop')
#sys.path.append('C:/Users/Brandon/Documents/ORISE/')
import tkinter
import itertools, math
import random
import matplotlib.pyplot as plt
import data_collection, predict, post_process, preprocess

"""visualize.py: Model visualization. Makes learning curves and feature curves
Example usage:
import visualize
visualize.learningCurve()# make learning curve
visualize.AICCurve()#make features curves

Changes from 0.0:
  Set xlim and ylim for plots with standard axes
  Added histogram plotter
  Added box and whisker plotter
Future changes:
  Parallelize learningCurve and AICCurve functions
  Rename AICCurve
"""
__author__    = "Brandon Veber"
__email__     = "veber001@umn.edu"
__version__   = "0.1"
__date__      = "7/25/2015"
__credits__   = ["Brandon Veber", "Rogelio Tornero-Velez", "Brandall Ingle",
                 "John Nichols"]
__status__    = "Development"

def learningCurve(featSelect='drugs',nFeatures=20,modelType='SVR',stdev=False,preSplit=False,
                  verbose=1):
    """Takes and splits data, then plots the learning curves.
    Inputs:
      drugs: dict, required. 'X' key is input features,'y' key is fraction unbound target value
      toxcast: dict, required. 'X' key is input features,'y' key is fraction unbound target value
      modelType: string, optional (default='SVR'). The selectable machine learing algorithm.
      verbose: int, optional (default=1). The verbosity of output statements
    """
    #Number of training samples to test
    trainingSampleSize = np.append([3,10,25,50],np.arange(100,1001,300))
    #Initialize output error dictionary
    rmse = {'train':{},'drugs':{},'toxcast':{}}
    #End index if shuffle = False. 
    nSims = 1
    #Iterate through training sample sizes
    for n in trainingSampleSize:
        #initialize arrays 
        rmse['train'][n]=[];rmse['drugs'][n]=[];rmse['toxcast'][n]=[]
        if verbose > 0:print(n,' samples being tested for learning curve')
        #Set index if Shuffle = True to perform 10 random resamplings
        if stdev: nSims=10
        #Iterate through random resamplings
        for i in range(1,nSims+1):
            if verbose > 1: print('Shuffle #',i)
            #Calculate predictions with unique random sampling of drugs training/test set
            preds,actuals = predict.main(featSelect,nFeatures,modelType=modelType,numTrainingSamples=n,
                                 feat_random_state=i,preSplit=preSplit,verbose=verbose)
            #Calculate result metrics and residual errors
            results,residuals = post_process.getResults(preds,actuals)
            #Append iteration to output error dictionary
            rmse['train'][n].append(results['Train']['rmse'])
            rmse['drugs'][n].append(results['Drugs']['rmse'])
            rmse['toxcast'][n].append(results['Toxcast']['rmse'])
        if verbose > 1:
            print(np.mean(rmse['train'][n]),np.mean(rmse['drugs'][n]),np.mean(rmse['toxcast'][n]),'\n')
    #Plot learning curve by sending errors to learningCurvePlotter module
    plot3(rmse,'Learning Curve','Number of training samples','RMSE',stdev,[0,.6],[0,1000])

def plot3(values,title='',xlabel='X',ylabel='y',stdev=False,ylim=None,xlim=None):
    plt.figure()
    colors = 'rgb'
    for i,key in enumerate(['toxcast','drugs','train']):
        ns,means,stds=[],[],[]
        for row in list(values[key].items()):
            ns.append(row[0])
            means.append(np.mean(row[1]))
            stds.append(np.std(row[1]))
        res = np.array((ns,means,stds)).T
        res = res[res[:,0].argsort()]
        plt.plot(res[:,0],res[:,1],color=colors[i],label=key.title())
        if stdev: stdevPlotter(res,colors[i])
    plotInfo(title,xlabel,ylabel,3,ylim,xlim)

def plot1(values,title='',xlabel='X',ylabel='y',color='r',stdev=False,ylim=None,xlim=None):
    plt.figure()
    ns,means,stds=[],[],[]
    for row in list(values.items()):
        ns.append(row[0])
        means.append(np.mean(row[1]))
        stds.append(np.std(row[1]))
    
    res = np.array((ns,means,stds)).T
    res = res[res[:,0].argsort()]
    plt.plot(res[:,0],res[:,1],color)
    if stdev: stdevPlotter(res,color)
    plotInfo(title,xlabel,ylabel,1,ylim,xlim)

def plotInfo(title,xlabel,ylabel,nLines,ylim=None,xlim=None):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if nLines> 1: plt.legend(loc='best')
    if ylim: plt.ylim(ylim[0],ylim[1])
    if xlim: plt.xlim(xlim[0],xlim[1])
    plt.show(block=False)

def stdevPlotter(res,color):
    upper = res[:,1]+res[:,2]
    plt.plot(res[:,0],upper,color+'--')
    lower = res[:,1]-res[:,2]
    plt.plot(res[:,0],lower,color+'--')
    plt.fill_between(res[:,0],lower,upper,color=color,alpha=.2)

def AICCurve(modelType='SVR',featSelect='drugs',stdev=False,preSplit=False,
             verbose=0):
    drugs,toxcast = data_collection.getData()
    plotAICCurve(drugs,toxcast,'drugs',modelType,stdev,preSplit,verbose)
    return

def plotAICCurve(drugs,toxcast,featSelect='drugs',modelType='SVR',stdev=False,
                 preSplit=False,verbose=1):
    numFeatures = [1]+list(np.arange(5,31,5))+[40,75,100,150]
    #if preSplit: numFeatures = [1]+list(np.arange(5,51,5))+[75,100,142,165]
    #numFeatures = [5,20,50]
    RMSE = {'train':{},'drugs':{},'toxcast':{}}
    AIC = {'train':{},'drugs':{},'toxcast':{}}
    BIC = {'train':{},'drugs':{},'toxcast':{}}
    drugs_X = drugs['X'].copy()
    toxcast_X = toxcast['X'].copy()
    nSims=1
    colors = {'train':'b','drugs':'g','toxcast':'r'}
    if stdev: nSims=5
    for n in numFeatures:
        drugsTemp = {'y':drugs['y'].copy(),'X':np.array(drugs_X)}
        toxcastTemp = {'y':toxcast['y'].copy(),'X':np.array(toxcast_X)}
        if verbose > 0:print(n,' features being tested for AIC')
        allResults={'Train':{'rmse':[]},'Drugs':{'rmse':[]},'Toxcast':{'rmse':[]}}
        for key in AIC: AIC[key][n] = []
        for key in BIC: BIC[key][n] = []
        for key in RMSE: RMSE[key][n] = []
        for i in range(1,nSims+1):
            if verbose > 0: print('Random Feature State: ',i)
            preds,actuals= predict.main(featSelect,modelType=modelType,
                                        nFeatures=n,feat_random_state=i,
                                        verbose=verbose,preSplit=preSplit)
            results,residuals = post_process.getResults(preds,actuals)
            for key in AIC:
                AIC[key][n].append(post_process.calculateAIC(len(preds[key]),
                                                results[key.title()]['rmse'],n))
            for key in BIC:
                BIC[key][n].append(post_process.calculateBIC(len(preds[key]),
                                                results[key.title()]['rmse'],n)) 
            for key in RMSE:
                RMSE[key][n].append(results[key.title()]['rmse'])
        if verbose > 1:
            print('AIC',AIC['train'][n],'\n',AIC['drugs'][n],'\n',AIC['toxcast'][n],'\n')
            print('BIC',BIC['train'][n],'\n',BIC['drugs'][n],'\n',BIC['toxcast'][n],'\n')
    plot3(RMSE,'RMSE vs Num Features','Num Features','RMSE',stdev,[0,.35],[0,160])
    for key in AIC:
        plot1(AIC[key],'AIC '+key.title(),'Num Features','AIC',colors[key],stdev)
        plot1(BIC[key],'BIC '+key.title(),'Num Features','BIC',colors[key],stdev)

def plotReliabilityGaussian(pred,reliability):
    gaussian = post_process.gaussian(np.linspace(-3,3,120),reliability[0],reliability[1])
    gaussianLeft = pred-gaussian
    gaussianRight = pred+gaussian
    return(gaussianLeft,gaussianRight)

def plotHistogramResiduals(residuals,verbose=0):
    for key in residuals:
        if verbose > 0: print(key,'Mean: ',np.mean(residuals[key]),'STDev: ',np.std(residuals[key]))
        plt.figure()
        hist = plt.hist(residuals[key],25)
        plt.title('Histogram of Residuals for '+key.title()+' set')
        plt.xlim(-1,1)
        plt.show(block=False)
        #plt.savefig(key+'_hist')

def boxPlot(featSelect='predefined',numTrainingSamples=1045,nFeatures=10,modelType='SVR',
            random_state=1):

    trainingFile = 'C:/Users/Brandon/Documents/ORISE/drug_training_192.csv'
    testFile = 'C:/Users/Brandon/Documents/ORISE/drug_test_192.csv'
    toxcastFile = 'C:/Users/Brandon/Documents/ORISE/toxcast_phase2_data_descriptors.csv'#toxcast_test_192.csv
    train = data_collection.extractCSV(trainingFile,'MOE');trainC={'X':train['X'].copy()}
    test = data_collection.extractCSV(testFile,'MOE')
    toxcast = data_collection.extractCSV(toxcastFile,'MOE')
    random.seed(random_state)
    trainIndices = random.sample(range(len(train['y'])),numTrainingSamples)
    train['indices']=trainIndices;test['indices']=range(len(test['y']));toxcast['indices']=range(len(toxcast['y']))
##    train['X'] = np.array(train['X'])[trainIndices];train['y']=train['y'][trainIndices]
    #Scale input features
    trainN,testN,toxcastN,yscaler = preprocess.normalize(train,test,toxcast)
    #Find important features
    featureList = preprocess.findFeatures(trainN,nFeatures,featSelect,featTypes='RF')
    featureNames = np.array(trainC['X'].columns.values)[featureList]
    train['X']=train['X'][:,featureList];test['X']=test['X'][:,featureList];toxcast['X']=toxcast['X'][:,featureList]
    for i in range(len(train['X'][0])):
        plt.close(i+1)
        fig=plt.figure(i+1)
        ax = fig.add_subplot(111)
        data=[train['X'][:,i],test['X'][:,i],toxcast['X'][:,i]]
        merged = list(itertools.chain(*data))
        bp=ax.boxplot(data,labels=['Training','Test','Toxcast'],whis='range')
        top = np.max(merged)
        bottom = np.min(merged)
        ax.set_ylim(bottom-.1,top+.1)
        ax.set_title(featureNames[i])
##        ax.set_title('Feature #'+str(i+1))
        fig.savefig('Feature_'+str(i+1))
        plt.close(fig)
    
    plt.close(i+2)
    fig=plt.figure(i+2)
    ax = fig.add_subplot(111)
    data = [train['y'],test['y'],toxcast['y']]
    bp = ax.boxplot(data,labels=['Training','Test','Toxcast'],whis='range')
    ax.set_ylim(-.1,1.1)
    ax.set_title('Fraction Unbound')
    plt.show(block=False)
##    plt.close(fig)
##    fig.savefig('Fraction Unbound')
