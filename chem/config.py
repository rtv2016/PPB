import os

__author__ = "bveber"

path = os.path.dirname(__file__)
datapath = os.path.abspath(os.path.join(path, os.pardir, "data"))
TRAININGFILE = os.path.join(datapath, "drug_training_192.csv")
TESTFILES = {
    "drugs": os.path.join(datapath, "drug_test_192.csv"),
    "toxcast_phase_1": os.path.join(datapath, "toxcast_test_192_Phase_I.csv"),
    "toxcast_phase_2": os.path.join(datapath, "toxcast_test_192_Phase_II.csv"),
}
