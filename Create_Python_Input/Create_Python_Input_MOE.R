# Create_Python_input_MOE.R


# Temporary solution to  generate blind predictions for new compounds
# Using the model descrinde by Ingle etal
# Use package PPB-master_blind (predictions for chemicals w/o Fup values) 
# You will call your datafile 'toxcast_test_192_Phase_II.csv' so caution is required!
# 
# Run this script to create toxcast_test_192_Phase_II.csv 
# Retain original toxcast_test_192_Phase_II.csv (2/26/2016) in PPB-master_blind/data/hold directory
# Place new toxcast_test_192_Phase_II.csv in PPB-master_blind/data
#  
# In PPB-master_blind directory specify output file name for Predictions.py script
# output is same order as input file 


library(plyr)

#warning, set path accordingly
setwd("C:/Users/rtorne02/Desktop/Blind PP predictions/Create_Python_Input/Input")

ToxCast_192<- read.csv("toxcast_test_192_Phase_II.csv")
#ToxCast_MOE<- read.csv("ToxCast_MOE.csv") # Used for AER
Exponent_MOE <-read.csv("rtv_083019_2.csv") # Used for Exponent


SVMvars  <- c("logS","VAdjEq","PEOE_VSA_FPPOS","SlogP","a_nS","a_base",
              "a_nN","SlogP_VSA6","logP.o.w.","PEOE_VSA_FPOL")

RFvars  <- c("logS","logP.o.w.","SlogP","PEOE_VSA_FPPOS","GCUT_SMR_0",
             "BCUT_SLOGP_0","GCUT_PEOE_0","GCUT_PEOE_1","GCUT_SLOGP_0","PEOE_VSA_PPOS")

myvars <- union(SVMvars,RFvars)


####
# Jump to 1, 2, or 99(QC)
##########################################################
# 1 AER



headers <- colnames(ToxCast_192) 
columns <- ncol(ToxCast_192)
rows    <- nrow(ToxCast_MOE)
 
template <- mat.or.vec(rows, columns)
colnames(template) <- headers
dftemplate <- as.data.frame(template)


dftemplate$ID.     <- ToxCast_MOE$Study_ID       
dftemplate$SMILES  <- ToxCast_MOE$Structure
dftemplate$Fub    <- 0.0
dftemplate$CAS     <- ToxCast_MOE$ID

#descriptors

dftemplate$logS           <- ToxCast_MOE$logS
dftemplate$VAdjEq         <- ToxCast_MOE$VAdjEq 
dftemplate$PEOE_VSA_FPPOS <- ToxCast_MOE$PEOE_VSA_FPPOS
dftemplate$SlogP          <- ToxCast_MOE$SlogP
dftemplate$a_nS           <- ToxCast_MOE$a_nS
dftemplate$a_base         <- ToxCast_MOE$a_base
dftemplate$a_nN           <- ToxCast_MOE$a_nN
dftemplate$SlogP_VSA6     <- ToxCast_MOE$SlogP_VSA6
dftemplate$logP.o.w.      <- ToxCast_MOE$logP.o.w.
dftemplate$PEOE_VSA_FPOL  <- ToxCast_MOE$PEOE_VSA_FPOL
dftemplate$GCUT_SMR_0     <- ToxCast_MOE$GCUT_SMR_0
dftemplate$BCUT_SLOGP_0   <- ToxCast_MOE$BCUT_SLOGP_0
dftemplate$GCUT_PEOE_0    <- ToxCast_MOE$GCUT_PEOE_0
dftemplate$GCUT_PEOE_1    <- ToxCast_MOE$GCUT_PEOE_1
dftemplate$GCUT_SLOGP_0   <- ToxCast_MOE$GCUT_SLOGP_0
dftemplate$PEOE_VSA_PPOS  <- ToxCast_MOE$PEOE_VSA_PPOS

dftemplate <- rename(dftemplate, c("ID." = "ID#"))

dftemplate <- rename(dftemplate, c("logP.o.w." = "logP(o/w)"))

setwd("/home/rtorne02/Desktop/BrandonFuR_atom/PPB-master/data")
write.csv(dftemplate,"toxcast_test_192_Phase_II.csv",row.names=FALSE)

setwd("/home/rtorne02/Desktop/BrandonFuR_atom/PPB-master/")
write.csv(dftemplate,"toxcast_test_192_Phase_II.csv",row.names=FALSE)

#######################
#2 Exponent


headers <- colnames(ToxCast_192) 
columns <- ncol(ToxCast_192)
rows    <- nrow(Exponent_MOE)

template <- mat.or.vec(rows, columns)
colnames(template) <- headers
dftemplate <- as.data.frame(template)


dftemplate$ID.     <- Exponent_MOE$ID.       
dftemplate$SMILES  <- Exponent_MOE$SMILES
dftemplate$Fub    <- 0.0
dftemplate$CAS     <- Exponent_MOE$CAS
dftemplate$Name   <- Exponent_MOE$Name

#descriptors

dftemplate$logS           <- Exponent_MOE$logS
dftemplate$VAdjEq         <- Exponent_MOE$VAdjEq 
dftemplate$PEOE_VSA_FPPOS <- Exponent_MOE$PEOE_VSA_FPPOS
dftemplate$SlogP          <- Exponent_MOE$SlogP
dftemplate$a_nS           <- Exponent_MOE$a_nS
dftemplate$a_base         <- Exponent_MOE$a_base
dftemplate$a_nN           <- Exponent_MOE$a_nN
dftemplate$SlogP_VSA6     <- Exponent_MOE$SlogP_VSA6
dftemplate$logP.o.w.      <- Exponent_MOE$logP.o.w.
dftemplate$PEOE_VSA_FPOL  <- Exponent_MOE$PEOE_VSA_FPOL
dftemplate$GCUT_SMR_0     <- Exponent_MOE$GCUT_SMR_0
dftemplate$BCUT_SLOGP_0   <- Exponent_MOE$BCUT_SLOGP_0
dftemplate$GCUT_PEOE_0    <- Exponent_MOE$GCUT_PEOE_0
dftemplate$GCUT_PEOE_1    <- Exponent_MOE$GCUT_PEOE_1
dftemplate$GCUT_SLOGP_0   <- Exponent_MOE$GCUT_SLOGP_0
dftemplate$PEOE_VSA_PPOS  <- Exponent_MOE$PEOE_VSA_PPOS

dftemplate <- rename(dftemplate, c("ID." = "ID#"))

dftemplate <- rename(dftemplate, c("logP.o.w." = "logP(o/w)"))

setwd("C:/Users/rtorne02/Desktop/Blind PP predictions/Create_Python_Input/Input/Exponent")
write.csv(dftemplate,"toxcast_test_192_Phase_II.csv",row.names=FALSE)








# 99
#STOP !! unless doing QC

# QC of process with "toxcast_test_192_Phase_I.csv"
# This recreate toxcast_test_192_Phase_I.csv but with all non-necessary 
# descriptors replaced with 0's 
# ptyhon package gives the same fit using only necessary descriptors

headers <- colnames(ToxCast_192) 
columns <- ncol(ToxCast_192)
rows    <- nrow(ToxCast_192)
template <- mat.or.vec(rows, columns)

colnames(template) <- headers
dftemplate <- as.data.frame(template)
  
  dftemplate$ID.     <- ToxCast_192$ID.       
  dftemplate$SMILES  <- ToxCast_192$SMILES
  dftemplate$CAS     <- ToxCast_192$CAS
  dftemplate$Name    <- ToxCast_192$Name
  
  
  
#descriptors

  dftemplate$logS           <- ToxCast_192$logS
  dftemplate$VAdjEq         <- ToxCast_192$VAdjEq 
  dftemplate$PEOE_VSA_FPPOS <- ToxCast_192$PEOE_VSA_FPPOS
  dftemplate$SlogP          <- ToxCast_192$SlogP
  dftemplate$a_nS           <- ToxCast_192$a_nS
  dftemplate$a_base         <- ToxCast_192$a_base
  dftemplate$a_nN           <- ToxCast_192$a_nN
  dftemplate$SlogP_VSA6     <- ToxCast_192$SlogP_VSA6
  dftemplate$logP.o.w.      <- ToxCast_192$logP.o.w.
  dftemplate$PEOE_VSA_FPOL  <- ToxCast_192$PEOE_VSA_FPOL
  dftemplate$GCUT_SMR_0     <- ToxCast_192$GCUT_SMR_0
  dftemplate$BCUT_SLOGP_0   <- ToxCast_192$BCUT_SLOGP_0
  dftemplate$GCUT_PEOE_0    <- ToxCast_192$GCUT_PEOE_0
  dftemplate$GCUT_PEOE_1    <- ToxCast_192$GCUT_PEOE_1
  dftemplate$GCUT_SLOGP_0   <- ToxCast_192$GCUT_SLOGP_0
  dftemplate$PEOE_VSA_PPOS  <- ToxCast_192$PEOE_VSA_PPOS
  
  dftemplate <- rename(dftemplate, c("ID." = "ID#"))
  #dftemplate <- rename(dftemplate, c("Fu" = "Fub"))
  #dftemplate <- rename(dftemplate, c("CAS." = "CAS#"))
  dftemplate <- rename(dftemplate, c("logP.o.w." = "logP(o/w)"))

  
  setwd("C:/Users/rtorne02/Desktop/Blind PP predictions/Python_Input/Input/QC")  
  write.csv(dftemplate,"toxcast_test_192_Phase_II.csv",row.names=FALSE)
  
 
  
  
  