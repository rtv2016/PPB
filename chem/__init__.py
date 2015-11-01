import os,sys
#sys.path.append(os.path.realpath("__file__"))
# dirsToAdd = [dir[0] for dir in os.walk(os.path.dirname(os.path.realpath("__file__")))]
# print(dirsToAdd)
# sys.path=sys.path+dirsToAdd
from chem.data_collection import Collector
from chem.preprocess import Scaler
from chem.preprocess import Reducer
from chem.model import Modeler

__author__ = 'bveber'
