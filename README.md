# FX-Grader

## Dataset
I created a resampled version of the dataset to avoid long processing times and this data can be found in `/Spine_data/Data_processed/ASL_ISO/dataset-ctfu-latest` I recomment to download this to your local machine to improve data loading efficiency and avoid permission errors in some setups.  

## Installation instructions

to run this project, you would need to download and install anaconda / miniconda



### MAC OS / Linux
1) download and install miniconda / anaconda:https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html 
make sure you perform these steps by running in zsh as a default shell: `chsh -s /bin/zsh` and restart your terminal after the installation of miniconda.
2) clone the repository using the comman `git clone https://github.com/malekosh/FX-Grader.git`
if git is not installed, type in `brew install git`
3) navigate to the main repository directory (where the .ipynb and .yml files are) via command line
4) install conda environment by typing
```
conda env create -f environment.yml
```
5) activate the virtual environment by typing (You need to perform this step everytime before opening a new session)
```
conda activate grader_followup
```
or 
```
source activate grader_followup
```
6) run jupyter-lab by typing `jupyter-lab` in the command line and make sure that you are in the project directory
7) enjoy the supreme grading experience :D

#### Notes
if the command line does not recognize the conda command, then try activating the base environment at the beginning by typing
```
source activate base
```

### Windows
1) download and install anaconda: https://docs.anaconda.com/anaconda/install/windows/
2) from your anaconda home, open CMD.exe Prompt as shown in the figure below:

![Screenshot](./readme_figs/powershell.png?raw=true "Optional Title")


3) Navigate to the project directory by typing:
```
cd path\to\project
```
In the following figure I navigate to my project which can be found in the Documents folder, inside FX-Grader-Followup\FX-Grader-Followup:
![Screenshot](./readme_figs/cd.png?raw=true "Optional Title")


4) Install the project's evironment by typing (make sure you end up in the main directory of the repository)
```
conda env create -f environment.yml
```
This would take some time

5) verify that the new environment has been installed and switch to that environment from the anaconda navigator. You should find an entry called grader_followup in the envronment tab. Once you see it, click on it to activate it. 

![Screenshot](./readme_figs/activate.png?raw=true "Optional Title")

6) Go back to the home tab of the anaconda navigator and start jupyter lab.
![Screenshot](./readme_figs/jupyter.png?raw=true "Optional Title")

7) Navigate to you project via the folder manager on the left side:
![Screenshot](./readme_figs/navigate.png?raw=true "Optional Title")

### General Notes
From experience I found that the default white background of jupyter can become exhasting on the eyes, so I recommend to switch to the dark theme by pressing on the settings tab at the top left as follows:
![Screenshot](./readme_figs/theme.png?raw=true "Optional Title")

Finally, to ensure you make the most of your screen's width, I recommend to hide the folder navigator on the left side by pressing ctrl+b. The same command can be used to show it back again.