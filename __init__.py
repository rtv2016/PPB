import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
print(sys.path)

import data_collection
import preprocess
import predict
import post_process
import visualize

def main(save=False,verbose=0):
    return(predict.main(save=save,verbose=verbose))
