import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
print(sys.path)

import old.predict
from old import visualize, predict


def main(save=False,verbose=0):
    return(predict.main(save=save,verbose=verbose))
