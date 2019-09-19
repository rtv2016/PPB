# 9-19-2019 
# The PPB-master Python package was used to create and validate Ingle et al. 2016, but does not make predictions for new chemicals.
# new_predictions_sep19 is a temporary fix to generate predictions (‘blind’ for predictions without associated experimental Fup values). 
# As with PPB-master which uses MOE descriptors (https://www.chemcomp.com/), new_predictions_sep19 requires MOE descriptors (but not the full 192 set)
   
# These MOE descriptors are required
 [1] "logS"           "VAdjEq"         "PEOE_VSA_FPPOS" "SlogP"          "a_nS"          
 [6] "a_base"         "a_nN"           "SlogP_VSA6"     "logP.o.w."      "PEOE_VSA_FPOL" 
[11] "GCUT_SMR_0"     "BCUT_SLOGP_0"   "GCUT_PEOE_0"    "GCUT_PEOE_1"    "GCUT_SLOGP_0"  
[16] "PEOE_VSA_PPOS" 

# We recently discovered (8/30/2019) that sccording to MOE/CCG chemicals should be represented with explicit hydrogens. But this was not the case in the training, thus for consistency this should be off- no explicit hydrogen. Example MOE descriptor calculations are given in the Create_Python_Input folder for spefic SMILES strings. 

# Use  Create_Python_Input_MOE.R to create the input file for PPB-master_blind.
# This script is in the Create_Python_Input folder   
# Run this script to create a ‘new’ toxcast_test_192_Phase_II.csv file which will be used for predictions 
# Retain the original toxcast_test_192_Phase_II.csv (2/26/2016) in new_predictions_sep19/data/hold directory
# Place ‘new’ toxcast_test_192_Phase_II.csv in PPB-master_blind/data
# Important: in Predictions.py script, at end, specify a new output file name 
# The order of chemical predictions in output file is the same as in the toxcast_test_192_Phase_II.csv file
# The original toxcast_test_192_Phase_II.csv is 168 rows by 192 columns.  The new one can be any row size, but must be 197 columns.  
The R program replaces the 16 essential descriptors in the correct column positions.  The other 192-16 descriptors are replaced with 0’s.  
# IMPORTANT: if run is used for blind predictions (Fup=0), then MAE and RMSE are meaningless. Conversely, if Fup are available the MAE and RMSE measures are valid
	

