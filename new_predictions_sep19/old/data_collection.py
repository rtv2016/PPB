import pandas as pd
import numpy as np
import sys, tkinter
sys.path.append('C:/Users/Brandon/Documents/ORISE/')

"""data_collection.py: Collects molecular data from CSV and Excel files
Example usage:
import data_collection as dc
drugs,toxcast = dc.getData(None,None); #Now use GUI to select appropriate files

Changes for version_0.0:

Future Changes:
  Accept spreadsheet without fub column for blind predictions
"""
__author__    = "Brandon Veber"
__email__     = "veber001@umn.edu"
__version__   = "0.0"
__date__      = "7/14/2015"
__credits__   = ["Brandon Veber", "Rogelio Tornero-Velez", "Brandall Ingle",
                 "John Nichols"]
__status__    = "Development"


def getData(drugsFileName='C:/Users/Brandon/Documents/ORISE/DRUGS_DESCRIPTORS_MOE_ALL2D.csv',
            toxFileName =  'C:/Users/Brandon/Documents/ORISE/TOXCAST_DESCRIPTORS_MOE_ALL2D.csv',
            descriptorSet = 'MOE'):
    """Gets data from MOE csv files. If filenames aren't given, then a GUI
    will open, to choose the correct files.
    Inputs:
      drugsFileName: string, optional. 
      toxFileName: string, optional.
      descriptorSet: string, optional. describes CSV format. MOE or Dragon
    Outputs:
      drugs: dataframe. All data from MOE csv files drugs data set
      toxcast: dataframe.  All data from MOE csv files for toxcast data set
    """
    if not drugsFileName:
        drugsFileName= getFname('Drugs')
    if not toxFileName:
        toxFileName = getFname('Toxcast')
    drugs = extractCSV(drugsFileName,descriptorSet)
    toxcast = extractCSV(toxFileName,descriptorSet)
    return(drugs,toxcast)

def getFname(featureSet):
    """GUI for selecting the proper CSV files
    Inputs:
      featureSet: str, required.  This will give a message to the user to
      determine which set of molecular descriptors will be used.
    Outputs:
      fname: str.  The filename that is output based on the user selection
    """
    root = tkinter.Tk()
    fname = tkinter.filedialog.askopenfilename(title='Open '+featureSet+' file',filetypes=(('CSV Files','.csv'),('All Files','.*')))
    root.destroy()
    return(fname)

def extractCSV(filename,descriptorSet):
    """Extract data from .csv file
    Inputs:
      filename: string, required. The CSV file with descriptors and target value
      descriptorSet: string, required. describes CSV format. MOE or Dragon
    Outputs:
      dictionary: Keys are 'X' (molecular descriptors) and 'y' (target values)
    """
    #Indices for column location of target values in CSV
    indFu={'MOE':2,'dragon':1}
    #Indices for starting column location of descriptors in CSV
    indDesc={'MOE':5,'dragon':3}
    allData = pd.read_csv(filename)
    y = np.array(allData.ix[:,indFu[descriptorSet]])
    X = allData.ix[:,indDesc[descriptorSet]:]
    return({'X':X,'y':y})

def getIonicClass(fname):
    """Extract ionic data from excel file
    Inputs:
      fname: string, required. Excel file containing ionic information
    Outputs:
      array. Ionic classification """
    data = pd.read_excel(fname)
    return(np.array(data['Ionic']))
