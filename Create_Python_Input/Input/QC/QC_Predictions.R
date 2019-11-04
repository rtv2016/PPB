# Create_Python_input_MOE.R



# Coerce MOE file for Python Input
# In PPB Python package
# Call it toxcast_test_192_Phase_II_coerce.csv
# Retain real toxcast_test_192_Phase_II.csv in PPB/hold directory
# Delete toxcast_test_192_Phase_II.csv in PPB folder
# In PPB folder rename toxcast_test_192_Phase_II_coerce.csv to toxcast_test_192_Phase_II.csv 
# run test3.py or test4.py

load(file="QC.RData")

library(plyr)

setwd("C:/Users/rtorne02/Desktop/Blind PP predictions/Python_Input/Input/QC")

Fup.Supp             <- read.csv("Supp_PPB_JCIM_AllData.csv")
Fup.Blind.T2         <- read.csv("ToxCast_II_BlindMethod.csv")

myvars <- names(Fup.Blind.T2) %in% c("X")
Fup.Blind.T2 <- Fup.Blind.T2[!myvars]
Fup.Blind.T2$Con_py_pred <- rowMeans(Fup.Blind.T2[c('kNN_py_pred','SVM_py_pred', 'RF_py_pred' )], na.rm=TRUE)


Fup.Supp.T2 <- Fup.Supp[ which(Fup.Supp$Set=='T2' ), ]
 
Fup.predicts <-cbind(Fup.Supp.T2,Fup.Blind.T2)  
  
corr.consensus  <- cor(Fup.predicts$Con_Pred, Fup.predicts$Con_py_pred)
corr.knn        <- cor(Fup.predicts$kNN_Pred, Fup.predicts$kNN_py_pred)
corr.svm        <- cor(Fup.predicts$SVM_Pred, Fup.predicts$SVM_py_pred)
corr.rf         <- cor(Fup.predicts$RF_Pred, Fup.predicts$RF_py_pred)

save.image(file="QC.RData") 