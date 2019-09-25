import random
from collections import Counter
import numpy as np
import sklearn
from sklearn import model_selection, preprocessing, metrics, model_selection
import chem

"""preprocess.py: Scale feature set and target values, reduce features
"""
__author__ = "Brandon Veber"
__email__ = "veber001@umn.edu"
__date__ = "10/31/2015"
__credits__ = ["Brandon Veber", "Rogelio Tornero-Velez", "Brandall Ingle", "John Nichols"]
__status__ = "Development"


class lnKaScaler:
    def __init__(self):
        self.description = "Scale fraction unbound to pseudo-equilibrium constant"

    def fit(self, data):
        return self

    def fit_transform(self, values):
        y_temp = np.array([np.max([.001, np.min([.99, sample])]) for sample in values])
        return .5 * np.log(y_temp / (1 - y_temp))

    def transform(self, values):
        y_temp = np.array([np.max([.001, np.min([.99, sample])]) for sample in values])
        return .5 * np.log(y_temp / (1-y_temp))

    def inverse_transform(self, values):
        return np.exp(2 * values) / (1 + np.exp(2 * values))


class Scaler:
    xscalers = {'MinMax': preprocessing.MinMaxScaler(),
                'standard': preprocessing.StandardScaler(),
                None: None}
    yscalers = {'lnKa': lnKaScaler(),
                'standard': preprocessing.StandardScaler(),
                None: None}

    def __init__(self, featSelect='predefined_RF', featTypes='RF',
                 random_state=1, yscale='lnKa', xscale='MinMax', verbose=0):
        self.__dict__.update(**locals())
        self.imputer = preprocessing.Imputer()
        self.xscaler = self.xscalers[self.xscale]
        self.yscaler = self.yscalers[self.yscale]

    def fit(self, data):
        y = np.reshape(data['y'], (-1, 1))
        self.imputer.fit(data['X'])
        if self.scale is not None:
            self.xscaler.fit(data['X'])
        if self.yscale is not None:
            self.yscaler.fit(y)

    def fit_transform(self, data):
        y = np.reshape(data['y'], (-1, 1))
        X_scaled = self.imputer.fit_transform(data['X'])
        if self.xscale is not None:
            X_scaled = self.xscaler.fit_transform(X_scaled)
        if self.yscale is not None:
            y_scaled = self.yscaler.fit_transform(y)
        else:
            y_scaled = y
        return {'X': X_scaled, 'y': np.reshape(y_scaled, (len(y_scaled), ))}

    def transform(self, data):
        y = np.reshape(data['y'], (-1, 1))
        X_scaled = self.imputer.transform(data['X'])
        if self.xscale is not None:
            X_scaled = self.xscaler.transform(X_scaled)
        if self.yscale is not None:
            y_scaled = self.yscaler.transform(y)
        else:
            y_scaled = y
        return {'X': X_scaled, 'y': np.reshape(y_scaled, (len(y_scaled), ))}

    def inverse_transform(self, data):
        y = np.reshape(data['y'], (-1, 1))
        if self.xscale is None:
            X_orig = self.xscaler.inverse_transform(data['X'])
        else:
            X_orig = data['X']
        if self.yscaler is not None:
            y_orig = self.yscaler.inverse_transform(data['y'])
        else:
            y_orig = y
        return {'X': X_orig, 'y': np.reshape(y_orig, (len(y_orig), ))}


class Reducer:
    def __init__(self, yscaler = None, nFeatures=None, featSelect='drugs', featTypes='RF', modelType='RF',
                 nSims=10, verbose=0):
        self.__dict__.update(**locals())
        self.aic = None
        self.bic = None
        self.rmse = None
        self.featureList = None

    def fit(self, data):
        self.nFeatures, self.aic, self.bic, self.rmse = get_n_features(data, self.nFeatures, self.yscaler,
                                                                       self.featSelect, self.featTypes, self.modelType,
                                                                       self.nSims, self.verbose)
        self.featureList, self.featureRank = aggregate_feature_list(data, self.nFeatures, self.featSelect,
                                                                    self.featTypes, self.nSims)
        return self

    def transform(self, data):
        return data['X'][:, self.featureList]

    def fit_transform(self, data):
        self.nFeatures, self.aic, self.bic, self.rmse = get_n_features(data, self.nFeatures, self.yscaler,
                                                                       self.featSelect, self.featTypes, self.modelType,
                                                                       self.nSims, self.verbose)
        self.featureList, self.featureRank = aggregate_feature_list(data, self.nFeatures, self.featSelect,
                                                                    self.featTypes, self.nSims)
        return data['X'][:, self.featureList]

    def set_params(self, **kwargs):
        self.__dict__.update(**kwargs)


def aggregate_feature_list(data, nFeatures, featSelect, featTypes, nSims):
    featureListAgg = []
    for n in range(nSims):
        featureListAgg += list(find_features(data, nFeatures, featSelect, featTypes, n))
    featureList = [item[0] for item in Counter(featureListAgg).most_common(nFeatures)]
    featureRank = [item[1] for item in Counter(featureListAgg).most_common(nFeatures)]
    return featureList, featureRank


def get_n_features(data, nFeatures, yscaler, featSelect, featTypes, modelType, nSims, verbose):
    if not nFeatures:
        aic, bic, rmse = aic_curve(data, yscaler, featSelect, featTypes,
                                   modelType, nSims, verbose)
        nFeatures = min(aic, key=aic.get)
        return nFeatures, aic, bic, rmse
    else:
        return nFeatures, None, None, None


def aic_curve(train, yscaler=None, featSelect='drugs', featTypes='RF', modelType='RF', nSims=10, verbose=0):
    numFeatures = [1] + list(np.arange(5, len(train['X'][0]) / 10, 5)) + \
                  list(np.arange(len(train['X'][0]) / 10, len(train['X'][0]) / 4, 10)) + \
                  list(np.arange(len(train['X'][0]) / 4, len(train['X'][0]) / 2, 20))
    aic = {}
    bic = {}
    rmse = {}
    for n in numFeatures:
        n = int(n)
        model = chem.Modeler(modelType)
        if verbose > 0:
            print(n, ' features being tested')
            print(model.model)
        featureListAgg = []
        for i in range(nSims):
            featureListAgg += list(find_features(train, n, featSelect,featTypes, feat_random_state=i))
        featureList = [item[0] for item in Counter(featureListAgg).most_common(n)]
        featureRank = [item[1] for item in Counter(featureListAgg).most_common(n)]
        preds = model_selection.cross_val_predict(model, train['X'][:, featureList], train['y'], cv=5, n_jobs=-1)
        if yscaler:
            preds_unscaled = yscaler.inverse_transform(preds)
            actuals_unscaled = yscaler.inverse_transform(train['y'])
        else:
            preds_unscaled = preds
            actuals_unscaled = train['y']
        rmse[n] = np.sqrt(metrics.mean_squared_error(preds_unscaled, actuals_unscaled))
        aic[n] = calculate_aic(len(preds), rmse[n], n)
        bic[n] = calculate_bic(len(preds), rmse[n], n)
        if verbose > 1:
            print('AIC:', aic[n])
            print('BIC: ', bic[n])
            print('RMSE: ', rmse[n], '\n')
    return aic, bic, rmse


def calculate_aic(m, rmse, n):
    return m * np.log(100 * rmse ** 2) + 2 * (n + 1)


def calculate_bic(m, rmse, n):
    return m * np.log(100 * rmse ** 2) + n * np.log(m)


def find_features(data, nFeatures=10, featSelect='drugs',
                 featTypes='RF', feat_random_state=4):
    """Find the top features for a given dataset using LinearSVR Lasso and Random Forest
    Inputs
      data: matrix, n x d matrix (n = number of samples; d = number of features). Required
      nFeatures: int, number of features to keep. Optional (default = 25)
    Outputs
      featureList: list, n top features for each machine learning algorithm
    """
    # Transform data to numpy array
    if featSelect.split('_')[0] == 'predefined':
        return sub_features(featSelect.split('_')[1])
    elif featSelect == 'drugs':
        # Initialize Dictionaries
        clf,topFeatures = {}, {}
        endIndex = int(len(data['y']) / 2)
        random.seed(feat_random_state)
        ind = random.sample(range(len(data['y'])), len(data['y']))
        X = data['X'][ind]
        y = data['y'][ind]
        if featTypes not in ['SVR','RF','Lasso']: featTypes = 'RF'
        # if featTypes == 'SVR': featTypes = 'RF'
        # Recursive Feature Elimination for SVR
        if 'SVR' in featTypes:
            # orig: 4

            grid = model_selection.GridSearchCV(sklearn.svm.LinearSVR(random_state=feat_random_state),
                                            {'C': np.logspace(-2,0,3), 'epsilon': np.logspace(-2,0,3)  #,
                                             # 'loss':['epsilon_insensitive','squared_epsilon_insensitive']
                                             # }).fit(X[:endIndex],y[:endIndex])
                                            }).fit(X[:endIndex], y[:endIndex])
            #model = grid.best_estimator_;print(model)
            model = sklearn.svm.LinearSVR(C=.1, random_state=feat_random_state)
            clf['SVR'] = sklearn.feature_selection.RFE(model, nFeatures,10).fit(X[:endIndex], y[:endIndex])
            # clf['SVR'] = sklearn.feature_selection.RFE(sklearn.svm.LinearSVR(random_state=feat_random_state),
            #                                        nFeatures,10).fit(data['X'][:endIndex],data['y'][:endIndex])
            #Indices of top SVR features
            return np.where(clf['SVR'].ranking_ == 1)[0]
        # Recursive Feature Elimination for Lasso
        if 'Lasso' in featTypes:
            clf['Lasso'] = sklearn.feature_selection.RFE(sklearn.linear_model.Lasso(alpha=.001),
                                                   nFeatures).fit(X[:endIndex], y[:endIndex])
            # Indices of top Lasso features
            return np.where(clf['Lasso'].ranking_ == 1)[0]
        # Random Forest
        if 'RF' in featTypes:
            clf['RF'] = sklearn.ensemble.RandomForestRegressor(random_state=feat_random_state,
                                                               n_estimators=250).fit(X[:endIndex], y[:endIndex])
            # Indices of top RF features
            # print(np.argsort(clf['RF'].feature_importances_)[-nFeatures:])
            return np.argsort(clf['RF'].feature_importances_)[-nFeatures:]
            # return(np.sort(np.argsort(clf['RF'].feature_importances_)[-nFeatures:]))
    elif featSelect == None:
        return range(len(data['X'][0]))
    else:
        print('Incorrect feature selection type')

def sub_features(modelType):
    # featureList = 'a_donacc BCUT_SLOGP_0 BCUT_SLOGP_2 BCUT_SMR_2 BCUT_SMR_3 GCUT_PEOE_0 GCUT_SLOGP_0 GCUT_SLOGP_1 GCUT_SLOGP_2 logS PEOE_RPC- PEOE_VSA+0 PEOE_VSA+1 PEOE_VSA+3 PEOE_VSA-3 PEOE_VSA-4 PEOE_VSA_FNEG SlogP_VSA7 SMR_VSA2 SMR_VSA4'.split(' ')
    # featuresList = [65,73,87,175]#['GCUT_SLOGP_0', 'logS', 'PEOE_RPC-', 'PEOE_VSA+1']
    # featuresList = [15, 22, 29, 31, 33, 35, 56, 65, 66, 68, 69, 73, 87, 89, 94, 103, 120, 175,176, 178]
    if modelType == 'SVR': featuresList = [89, 175, 123, 154, 26, 8, 23, 161, 88, 121]
    else: featuresList = [89, 88, 154, 123, 73, 32, 65, 66, 69, 129]

    # featureList = ['logS']
    return featuresList

