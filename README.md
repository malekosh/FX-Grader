# FX-Grader

## Installation instructions

to run this project, you would need to download and install anaconda / miniconda



### MAC OS / Linux
1) download and install miniconda / anaconda:https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html
2) navigate to the main repository directory (where the .ipynb and .yml files are) via command line
3) install conda environment by typing
```
conda env create -f environment.yml
```
4) activate the virtual environment by typing
```
conda activate grader_followup
```
or 
```
source activate grader_followup
```
5) run jupyter-lab by typing `jupyter-lab` in the command line and make sure that you are in the project directory
6) enjoy the supreme grading experience :D

#### Notes
if the command line does not recognize the conda command, then try activating the base environment at the beginning by typing
```
source activate base
```

### Windows
1) download and install anaconda: https://docs.anaconda.com/anaconda/install/windows/
2) from your anaconda home, open CMD.exe Prompt as shown in the figure below:
![Screenshot](./readme_figs/powershell.png?raw=true "Optional Title")


3) install ipywidgets using conda prompt:
```
conda install -c conda-forge ipywidgets
```
4) Launch jupyterlab from Anaconda Navigator.

if you have installed the latest anacoda version this should be enough


### Ipywidgets installation documentation
If you still have trouble getting Ippwidgets running, please refer to the installation documentation: 

https://ipywidgets.readthedocs.io/en/latest/user_install.html


