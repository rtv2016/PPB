chem is a package that provides a structured methodology for QSAR regression modeling.
If predictions based on Ingle et al (2016) are of interest, the folder 'new_predictions_sep19' contains updates to allow for making Fup predictions based off of the trained model. 

Requires python 3.6+ and reasonably specific packages of pandas, numpy, scikit-learn ,etc. Using the anaconda environment.yml included in this repo will ensure that you have the right versions of these packages.

To install using anaconda:
* Clone this repository with a git tool (command line or GUI)
* 'git clone https://github.com/rtv2016/PPB.git'
* Navigate to the PPB root directory
* Create conda environment with 'conda env create -f environment.yml'
* you can then run 'conda env list' to confirm that the environment is there

To load the conda environment at any point thereafter
* Then run 'conda activate ppbenv'
* From the PPB directory and from inside the conda environment you can run 'python test.py' to confirm that it is working

Devs:
* Should additionally run 'conda install -c conda-forge black' for formatting

To create input file:
* TBD
* Update this process: Refer to new_predictions_sep19/Create_Python_Input for details    

	
conda install -c conda-forge black
