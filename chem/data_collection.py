import pandas as pd
import numpy as np
import sys
from tkinter import Tk,filedialog
from chem.config import TRAININGFILE, TESTFILES

"""data_collection.py: Collects molecular data from csv and Excel files
Example usage:
import chem
e = chem.Collector()
e.collect()

Future Changes:
  Accept spreadsheet without fub column for blind predictions
"""
__author__    = "Brandon Veber"
__email__     = "veber001@umn.edu"
__version__   = "0.0"
__date__      = "10/31/2015"
__credits__   = ["Brandon Veber", "Rogelio Tornero-Velez", "Brandall Ingle",
                 "John Nichols"]
__status__    = "Development"


class Collector:
    def __init__(self,inputTrainingFile=None,inputTestFiles=None,targetIndex=2,descriptorIndex=5):
        self.inputTrainingFile = inputTrainingFile
        self.inputTestFiles = inputTestFiles
        self.targetIndex = targetIndex
        self.descriptorIndex = descriptorIndex

    def collect(self):
        if self.inputTrainingFile:
            train = extract_csv(self.inputTrainingFile,self.targetIndex,self.descriptorIndex)
        elif TRAININGFILE:
            train = extract_csv(TRAININGFILE,self.targetIndex,self.descriptorIndex)
        else:
            train = extract_csv(get_fname("Training"),self.targetIndex,self.descriptorIndex)

        if self.inputTestFiles:
            test = get_test(self.inputTestFiles,self.targetIndex,self.descriptorIndex)
        elif TESTFILES:
            test = get_test(TESTFILES,self.targetIndex,self.descriptorIndex)
        else:
            test = extract_csv(get_fname("Test"),self.targetIndex,self.descriptorIndex)
        return train, test


def get_test(testFiles,targetIndex=2,descriptorIndex=5):
    test = {}
    if isinstance(testFiles,dict):
        for key in testFiles:
            test[key] = extract_csv(testFiles[key],targetIndex,descriptorIndex)
    elif isinstance(testFiles,str):
        test['test'] = extract_csv(testFiles,targetIndex,descriptorIndex)
    return test


def get_data(drugsFileName='C:/Users/Brandon/Documents/ORISE/DRUGS_DESCRIPTORS_MOE_ALL2D.csv',
            toxFileName =  'C:/Users/Brandon/Documents/ORISE/TOXCAST_DESCRIPTORS_MOE_ALL2D.csv',
            descriptorSet = 'MOE'):
    """Gets data from MOE csv files. If filenames aren't given, then a GUI
    will open, to choose the correct files.
    Inputs:
      drugsFileName: string, optional. 
      toxFileName: string, optional.
      descriptorSet: string, optional. describes csv format. MOE or Dragon
    Outputs:
      drugs: dataframe. All data from MOE csv files drugs data set
      toxcast: dataframe.  All data from MOE csv files for toxcast data set
    """
    if not drugsFileName:
        drugsFileName= get_fname('Drugs')
    if not toxFileName:
        toxFileName = get_fname('Toxcast')
    drugs = extract_csv(drugsFileName,descriptorSet)
    toxcast = extract_csv(toxFileName,descriptorSet)
    return drugs,toxcast


def get_fname(featureSet=""):
    """GUI for selecting the proper csv files
    Inputs:
      featureSet: str, required.  This will give a message to the user to
      determine which set of molecular descriptors will be used.
    Outputs:
      fname: str.  The filename that is output based on the user selection
    """
    root = Tk()
    root.update()
    fname = filedialog.askopenfilename(title='Open '+featureSet+' file',filetypes=(('csv Files','.csv'),('All Files','.*')))#Button(root, text=featureSet, command = open_file).pack(fill=X)
    root.destroy()
    #root.mainloop()
    return fname


def extract_csv(filename,targetIndex=2,descriptorIndex=5):
    """Extract data from .csv file
    Inputs:
      filename: string, required. The csv file with descriptors and target value
      descriptorSet: string, required. describes csv format. MOE or Dragon
    Outputs:
      dictionary: Keys are 'X' (molecular descriptors) and 'y' (target values)
    """
    allData = pd.read_csv(filename)
    y = np.array(allData.ix[:,targetIndex])
    X = allData.ix[:,descriptorIndex:]
    return {'X':X,'y':y}


def get_ionic_class(fname):
    """Extract ionic data from excel file
    Inputs:
      fname: string, required. Excel file containing ionic information
    Outputs:
      array. Ionic classification """
    data = pd.read_excel(fname)
    return np.array(data['Ionic'])
