import chem
import numpy as np

train, test = chem.Collector().collect()
scaler = chem.Scaler()
train_scaled = scaler.fit_transform(train)
for key in train_scaled:
    np.savetxt('train_scaled_' + key + '.csv', train_scaled[key], delimiter=',')
test_scaled = {}
for key in test:
    test_scaled[key] = scaler.transform(test[key])
    for key2 in test_scaled[key]:
        np.savetxt('test_scaled_' + key + '_' + key2 + '.csv', test_scaled[key][key2], delimiter=',')

