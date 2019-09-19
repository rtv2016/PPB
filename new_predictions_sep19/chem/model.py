import numpy as np
import warnings
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from sklearn.grid_search import GridSearchCV
import inspect

"""model.py: Model evaluation and prediction
"""
__author__ = "Brandon Veber"
__email__ = "veber001@umn.edu"
__version__ = "0.1"
__date__ = "10/31/2015"
__credits__ = ["Brandon Veber", "Rogelio Tornero-Velez", "Brandall Ingle", "John Nichols"]
__status__ = "Development"


class Modeler:
    def __init__(self, modelType='RF', random_state=None, verbose=0):
        self.__dict__.update(**locals())
        # Model Creation
        estimator = getEstimator(self.modelType)
        modelParams = getModelParams(self.modelType, estimator)
        if self.modelType in ['RF', 'KNN']:
	 
            self.model = getEstimator(self.modelType, self.random_state)
        else:
            self.model = GridSearchCV(estimator, modelParams, scoring='mean_squared_error', cv=3,
                                      verbose=self.verbose, n_jobs=-1)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        args, varargs, kw, default = inspect.getargspec(init)
        if varargs is not None:
            raise RuntimeError("scikit-learn estimators should always "
                               "specify their parameters in the signature"
                               " of their __init__ (no varargs)."
                               " %s doesn't follow this convention."
                               % (cls, ))
        # Remove 'self'
        # XXX: This is going to fail if the init is a staticmethod, but
        # who would do this?
        args.pop(0)
        args.sort()
        return args

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep: boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils/__init__.py but it gets overwritten
            # when running under python3 somehow.
            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if len(w) and w[0].category == DeprecationWarning:
                    # if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)

            # XXX: should we rather test if instance of estimator?
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out


def getEstimator(modelType, random_state=0):
    estimator = None
    if modelType in ['libsvm_rbf', 'libsvm_lin', 'SVR']:
        estimator = SVR()
    if modelType == 'KNN':
        estimator = KNeighborsRegressor(n_neighbors=10, leaf_size=30,  algorithm='auto', n_jobs=-1)
    if modelType == 'RF':
        estimator = RandomForestRegressor(n_estimators=250, random_state=random_state, n_jobs=-1)

    return estimator


def getModelParams(modelType, estimator):
    modelParams = None
    if modelType == 'SVR':
        modelParams = {'kernel': ['rbf'], 'C': [10, 50], 'epsilon': np.logspace(-1, 0, 3), 'cache_size': [4096]}
    if modelType == 'KNN':
        #modelParams = {'n_neighbors': [2, 10, 25], 'leaf_size': [1, 10], 'algorithm': ['auto', 'ball_tree']}
       modelParams = {'n_neighbors': [10], 'leaf_size': [1, 10], 'algorithm': ['auto', 'ball_tree']}

    if modelType == 'RF':
        modelParams = {'n_estimators': [500], 'n_jobs': [-1], 'max_features': ['auto'], 'min_samples_split': [50]}
    if modelType == 'Adaboost':
        modelParams = {'n_estimators': [25, 50, 75, 100], 'learning_rate': np.arange(.05, .15, .02),
                       'loss': ['linear', 'square', 'exponential']}
    return modelParams
