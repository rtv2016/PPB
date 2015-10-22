__author__ = 'bveber'
import sys, os
sys.path.append(os.getcwd())
import predict

trainingFile='/Users/bveber/chem/data/drug_training_192.csv'
testFile ='/Users/bveber/chem/data/drug_test_192.csv'
toxcastFile='/Users/bveber/chem/data/toxcast_test_192_Phase_II.csv'
r = predict.main(trainingFile,testFile,toxcastFile,verbose=1)

