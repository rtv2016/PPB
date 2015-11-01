import chem
from sklearn import metrics
import numpy as np

modelType = 'RF'
featTypes = 'RF'
train,test = chem.Collector().collect()
scaler = chem.Scaler()
train_scaled = scaler.fit_transform(train)
test_scaled = {}
for key in test:
    test_scaled[key] = scaler.transform(test[key])
reducer = chem.Reducer(yscaler=scaler.yscaler, nFeatures=10, featSelect='drugs', featTypes=featTypes,
                       modelType=modelType, nSims=25, verbose=2)
train_scaled_reduced = reducer.fit_transform(train_scaled)
print(train['X'].columns.values[reducer.featureList], reducer.featureList)
print(reducer.featureRank)
test_scaled_reduced = {}
for key in test:
    test_scaled_reduced[key] = reducer.transform(test_scaled[key])
results = {}
for key in test_scaled_reduced:
    results[key] = []
for i in range(50):
    model = chem.Modeler(modelType, random_state=i, verbose=1).fit(train_scaled_reduced, train_scaled['y'])
    for key in test_scaled_reduced:
        preds = model.predict(test_scaled_reduced[key])
        mae = metrics.mean_absolute_error(scaler.yscaler.inverse_transform(preds), test[key]['y'])
        results[key].append(mae)
for key in results:
    print(key, np.mean(results[key]), np.std(results[key]))